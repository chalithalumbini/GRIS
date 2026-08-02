# matcher.py  —  GRIS-DepScore v2.4 (AUX modal fix)
#
# Design principles:
#   - Edge = (head, dependent, relation) triple — not bare token
#   - Hungarian optimal matching (not greedy)
#   - Multiplicative scoring: dep is primary, head is confidence multiplier
#   - Bonus-only at edge level (no subtractive structural penalties)
#   - Language-adaptive blend: W*edge_F_beta + (1-W)*sent_cos
#   - Sentence-level penalties: negation mismatch (×0.90), PROPN check (×0.80)
#
# CHANGE v2.4:
#   AUX modal dependents of VERB heads included in extract_edges().
#   Previously STOP_UPOS excluded ALL AUX tokens as dependents, making
#   modal verbs ("kann", "muss", "soll", "wird") completely invisible.
#   Evidence: bathtub example (MQM=-10) scored 0.875 because the only
#   structural error — missing "kann" — was never extracted as an edge.
#   Fix: if tok.upos==AUX and deprel in {aux, aux:pass} and head.upos==VERB
#        → include the edge. Other AUX positions (copula etc) unchanged.
#
# CHANGE v2.3 (Evaluate.py):
#   parse_lang/task_lang split for zh→en direction.
#
# CHANGE v2.2:
#   DE STRUCT_WEIGHT raised 0.80 → 0.92.
#
# CHANGE v2.1:
#   (a) F_beta(1.5) replaces symmetric F1.
#   (b) Verbosity penalty (VP_THRESHOLD=1.4, VP_FACTOR=0.92).
#
# Every other constant is backed by WMT22 ablation evidence.
#
# Design principles:
#   - Edge = (head, dependent, relation) triple — not bare token
#   - Hungarian optimal matching (not greedy)
#   - Multiplicative scoring: dep is primary, head is confidence multiplier
#   - Bonus-only at edge level (no subtractive structural penalties)
#   - Language-adaptive blend: W*edge_F1 + (1-W)*sent_cos
#   - Only two penalties: negation mismatch (×0.90), PROPN subject check (×0.80)
#
# CHANGE v2.3 (Evaluate.py):
#   parse_lang / task_lang split: for zh→en, parse_lang="en" (text language)
#   while task_lang="zh" controls STRUCT_WEIGHT lookup. Fixes Stanza being
#   called with lang="zh" on English hypothesis/reference text.
#
# CHANGE v2.2:
#   DE STRUCT_WEIGHT raised 0.80 → 0.92.
#   With F_beta(1.5) in place, NO_STRUCTURE (0.2874) > FULL_V2 (0.2775) for DE —
#   sentence cosine is diluting the stronger F_beta edge signal. W=0.92 reduces
#   cosine contribution to 8%, preserving system-level smoothing (FULL_V2
#   system 0.5737 >> NO_STRUCTURE 0.4593) while recovering segment-level gain.
#
# CHANGE v2.1:
#   (a) F_beta(1.5) replaces symmetric F1. ABL_C WMT22 DE evidence: +0.045.
#   (b) Verbosity penalty (VP_THRESHOLD=1.4, VP_FACTOR=0.92) prevents system-level
#       collapse from recall inflation in verbose MT systems.
#
# Every other constant is backed by WMT22 ablation evidence.
#
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# ── Embedding floor ────────────────────────────────────────────────────────────
# Applied to dep_sim only. Revert-B ablation (floor=0.20) scored WORSE (0.198
# vs 0.226), confirming 0.30 is the right value for DE MQM correlation.
EMB_FLOOR = 0.30

# ── Head confidence multiplier ─────────────────────────────────────────────────
# score = dep_floored * (HEAD_BASE + HEAD_SCALE * head_sim01)
# Revert-A ablation (additive 0.5/0.5) scored WORSE (0.191 vs 0.226),
# confirming the multiplicative dep-primary design is correct.
HEAD_BASE  = 0.70
HEAD_SCALE = 0.30   # HEAD_BASE + HEAD_SCALE = 1.0

# ── Bonus gating threshold ────────────────────────────────────────────────────
# Structural bonuses only fire when base_sim >= this value.
# Revert-E ablation (ungated) scored WORSE (0.191 vs 0.226).
BONUS_SEM_THRESHOLD = 0.82

# ── Edge-level bonuses ─────────────────────────────────────────────────────────
BONUS_DEPREL_MATCH    = 0.12   # exact deprel label match
BONUS_EXACT_LEMMA     = 0.10   # identical lemma (always fires, no threshold)
BONUS_VERB_PARAPHRASE = 0.05   # both VERB + high head cosine
BONUS_CORE_ARG_MATCH  = 0.05   # both edges are core arguments

# ── Negation penalty ───────────────────────────────────────────────────────────
NEG_SENT_PENALTY = 0.90   # mismatch-only, fires once per sentence

# ── PROPN subject mismatch ────────────────────────────────────────────────────
MAIN_SUBJ_PROPN_PENALTY = 0.80   # sentence-level multiplier for wrong named entity

# ── Language-adaptive structural weight ───────────────────────────────────────
# score = W * edge_F1 + (1-W) * sent_cos
#
# Calibrated from WMT22 three-language experiments:
#   DE: Updated to W=0.92 (was 0.80).
#       Evidence: with F_beta(1.5) edge scoring, FULL_V2 (W=0.80) scores 0.2775
#       while NO_STRUCTURE scores 0.2874 — sentence cosine is net-harmful at
#       segment level for DE. Raising W reduces cosine dilution of the F_beta
#       signal. W=1.0 is avoided to preserve the small system-level smoothing
#       benefit the cosine provides (FULL_V2 system=0.5737 > NO_STRUCTURE=0.4593).
#       A small residual cosine weight (0.08) retains that benefit.
#   ZH: v3 ZH_SENT_BLEND=0.75 (W=0.25) gave full=0.265 → confirmed
#   RU: W=0.30 improved +0.015 over W=1.0 → lowered to 0.20 for next run
#
# Rule: lower W when no_struct > full (structure hurts)  ← currently DE
#       higher W when full > no_struct (structure helps)
STRUCT_WEIGHT = {
    "en": 0.75,
    "de": 0.92,   # raised from 0.80: cosine dilutes F_beta(1.5) for DE (no_struct > full_v2)
    "ru": 0.20,   # free word order + lexical MQM errors dominate
    "fr": 0.70,
    "es": 0.70,
    "fi": 0.65,
    "ar": 0.55,
    "tr": 0.50,
    "zh": 0.25,   # confirmed by v3 ZH_SENT_BLEND=0.75 giving Spearman=0.265
    "ja": 0.20,
}
STRUCT_WEIGHT_DEFAULT = 0.60

# Back-compat aliases used by dashboard and scorer
RESCUE_FACTOR         = 0.82
LOW_PARSER_CONFIDENCE = {"zh", "ja"}
ZH_SENT_BLEND         = 1.0 - STRUCT_WEIGHT["zh"]   # = 0.75

# ── Core argument relation set ─────────────────────────────────────────────────
CORE_ARGS = {"nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp", "nsubj:pass"}

# ── Stop POS tags ──────────────────────────────────────────────────────────────
STOP_UPOS = {"DET", "AUX", "PART", "SCONJ", "CCONJ", "INTJ", "PUNCT"}

# ── Voice-compatible relation pairs ───────────────────────────────────────────
VOICE_COMPATIBLE = {
    frozenset({"nsubj",      "obl:agent"}),
    frozenset({"nsubj",      "obl"}),
    frozenset({"obj",        "nsubj:pass"}),
    frozenset({"obj",        "nsubj"}),
}

# ── Content POS for sentence cosine ───────────────────────────────────────────
_CONTENT_UPOS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"}


# ══════════════════════════════════════════════════════════════════════════════
# Maths helpers
# ══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(a, b) -> float:
    if a is None or b is None:
        return 0.0
    if torch.all(a == 0) or torch.all(b == 0):
        return 0.0
    return float(F.cosine_similarity(a, b, dim=0).item())


def to_01(sim: float) -> float:
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))


def smooth_floor(sim01: float) -> float:
    if sim01 <= EMB_FLOOR:
        return 0.0
    return (sim01 - EMB_FLOOR) / (1.0 - EMB_FLOOR)


def _deprel_bonus(hyp_rel: str, ref_rel: str) -> float:
    if hyp_rel == ref_rel:
        return BONUS_DEPREL_MATCH
    if frozenset({hyp_rel, ref_rel}) in VOICE_COMPATIBLE:
        return 0.10
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Linguistic detection helpers
# ══════════════════════════════════════════════════════════════════════════════

_NEG_LEMMAS = {
    "not","no","never","neither","nor",
    "nicht","nie","kein","keine","keinen","keinem","keiner","keines","nichts","weder","ohne",
    "не","нет","ни","никогда",
    "ne","pas","jamais","rien","aucun",
    "no","nunca","jamás","tampoco",
}


def sentence_has_negation(dep_parse) -> bool:
    for t in getattr(dep_parse, "tokens", []) or []:
        if (getattr(t, "deprel", "") or "").lower() == "neg":
            return True
        lem = (getattr(t, "lemma", None) or "").lower().strip()
        txt = (getattr(t, "text",  None) or "").lower().strip()
        if lem in _NEG_LEMMAS or txt in _NEG_LEMMAS:
            return True
    return False


def sentence_has_passive(dep_parse) -> bool:
    """Attribution only — does not penalise score."""
    tokens = getattr(dep_parse, "tokens", []) or []
    if not tokens:
        return False
    id2 = {t.id: t for t in tokens}
    children = {}
    for t in tokens:
        hid = getattr(t, "head", 0) or 0
        if hid:
            children.setdefault(hid, []).append(t.id)
    for hid, deps in children.items():
        head = id2.get(hid)
        if head is None:
            continue
        if (getattr(head, "upos", "") or "X").upper() not in ("VERB", "AUX"):
            continue
        feats = getattr(head, "feats", None) or {}
        if isinstance(feats, dict) and feats.get("Voice") == "Pass":
            return True
        for did in deps:
            rel = (getattr(id2[did], "deprel", "") or "").lower()
            if rel in ("aux:pass", "nsubj:pass", "csubj:pass"):
                return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Edge extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_edges(dep_parse):
    """
    Extract dependency edges for scoring.

    AUX dependents of VERB heads are now included (modals: kann, muss, soll,
    tense auxiliaries: hat, wird). Previously excluded via STOP_UPOS, causing
    modal errors (missing "kann"/"muss") to be completely invisible to DepScore.
    Evidence: bathtub example MQM=-10, DepScore=0.875 because "kann" was skipped.

    Rule: if tok.upos == AUX and deprel in {aux, aux:pass} and head is VERB → include.
    All other AUX tokens remain excluded (e.g. copula "ist" as head).
    """
    id2tok = {t.id: t for t in dep_parse.tokens}
    edges = []
    for tok in dep_parse.tokens:
        if not tok.head or tok.head == 0:
            continue

        upos = (getattr(tok, "upos", "") or "X").upper()
        rel  = (getattr(tok, "deprel", "") or "dep").lower()

        # AUX modal: include if it is a modal/tense aux of a VERB head
        if upos == "AUX" and rel in ("aux", "aux:pass"):
            head_tok = id2tok.get(tok.head)
            if head_tok and (getattr(head_tok, "upos", "") or "X").upper() == "VERB":
                edges.append((head_tok, tok))
                continue
            # else: AUX in other positions → skip (fall through to STOP_UPOS)

        if upos in STOP_UPOS:
            continue
        head_tok = id2tok.get(tok.head)
        if not head_tok:
            continue
        if (getattr(head_tok, "upos", "") or "X").upper() in STOP_UPOS \
                and (getattr(head_tok, "upos", "") or "X").upper() not in {"AUX"}:
            continue
        edges.append((head_tok, tok))
    return edges


# ══════════════════════════════════════════════════════════════════════════════
# Content-lemma cosine (sentence-level semantic floor)
# ══════════════════════════════════════════════════════════════════════════════

def _content_lemma_text(dep_parse) -> str:
    """
    Join content-word lemmas for sentence cosine computation.
    Revert-F ablation (full sentence) scored WORSE (0.202 vs 0.226),
    confirming content-lemma filtering is the right approach.
    """
    tokens = getattr(dep_parse, "tokens", []) or []
    lemmas = []
    for t in tokens:
        if (getattr(t, "upos", "") or "").upper() not in _CONTENT_UPOS:
            continue
        lem = (getattr(t, "lemma", None) or getattr(t, "text", "") or "").lower().strip()
        if lem:
            lemmas.append(lem)
    return " ".join(lemmas) if lemmas else " ".join(t.text.lower() for t in tokens)


# ══════════════════════════════════════════════════════════════════════════════
# PROPN subject check (sentence-level named entity guard)
# ══════════════════════════════════════════════════════════════════════════════

def _get_root_nsubj(dep_parse):
    tokens = getattr(dep_parse, "tokens", []) or []
    root = next((t for t in tokens
                 if (getattr(t, "deprel", "") or "").lower() == "root"), None)
    if root is None:
        return None, None
    nsubj = next(
        (t for t in tokens
         if t.head == root.id
         and (getattr(t, "deprel", "") or "").lower().startswith("nsubj")),
        None
    )
    return root, nsubj


def _apply_main_subject_check(base, dep_h, dep_r, hyp_embeds, ref_embeds, embedder):
    """
    Scan all nsubj edges for PROPN tokens.
    If the worst-matching PROPN nsubj pair is below 0.85,
    apply MAIN_SUBJ_PROPN_PENALTY (×0.80) to the sentence score.
    Note: NE edge-level penalty removed in v2. Only sentence-level PROPN check remains.
    Handles multi-sentence inputs where the named entity is not the ROOT nsubj.
    """
    try:
        h_tokens = {t.id: t for t in (getattr(dep_h, "tokens", []) or [])}
        r_tokens = {t.id: t for t in (getattr(dep_r, "tokens", []) or [])}

        h_propn = [t for t in h_tokens.values()
                   if (getattr(t, "deprel", "") or "").lower().startswith("nsubj")
                   and (getattr(t, "upos", "") or "") == "PROPN"]
        r_propn = [t for t in r_tokens.values()
                   if (getattr(t, "deprel", "") or "").lower().startswith("nsubj")
                   and (getattr(t, "upos", "") or "") == "PROPN"]

        if not h_propn or not r_propn:
            return base

        worst_sim = 1.0
        for ht in h_propn:
            vh = (hyp_embeds or {}).get(ht.id) if embedder.embedding_type != "word" \
                 else embedder.get_word_embedding(ht.text)
            if vh is None:
                continue
            best = 0.0
            for rt in r_propn:
                vr = (ref_embeds or {}).get(rt.id) if embedder.embedding_type != "word" \
                     else embedder.get_word_embedding(rt.text)
                if vr is None:
                    continue
                best = max(best, to_01(cosine_similarity(vh, vr)))
            worst_sim = min(worst_sim, best)

        if worst_sim < 0.85:  # PROPN subject similarity threshold
            return base * MAIN_SUBJ_PROPN_PENALTY
    except Exception:
        pass
    return base


# ══════════════════════════════════════════════════════════════════════════════
# Core edge similarity
# ══════════════════════════════════════════════════════════════════════════════

def compute_edge_similarity(
    hyp_edge, ref_edge, hyp_embeds, ref_embeds, embedder,
    debug: bool = False, lang: str = "en",
):
    """
    Score one (hyp_edge, ref_edge) pair.

    Formula
    -------
    dep_floored = smooth_floor(to_01(cos(hyp_dep, ref_dep)))
    head_conf   = HEAD_BASE + HEAD_SCALE * to_01(cos(hyp_head, ref_head))
    base_sim    = dep_floored * head_conf

    Bonuses (gated on base_sim >= BONUS_SEM_THRESHOLD, except exact_lemma):
      +BONUS_DEPREL_MATCH     deprel match or voice alternation
      +BONUS_VERB_PARAPHRASE  both VERB + head cosine > 0.70
      +BONUS_CORE_ARG_MATCH   both in CORE_ARGS
      +BONUS_EXACT_LEMMA      identical lemma (always fires)

    No edge-level penalties in v2 design.
    """
    hyp_head, hyp_tok = hyp_edge
    ref_head, ref_tok = ref_edge

    if embedder.embedding_type == "word":
        v_h_head = embedder.get_word_embedding(hyp_head.text)
        v_h_dep  = embedder.get_word_embedding(hyp_tok.text)
        v_r_head = embedder.get_word_embedding(ref_head.text)
        v_r_dep  = embedder.get_word_embedding(ref_tok.text)
    else:
        v_h_head = hyp_embeds.get(hyp_head.id)
        v_h_dep  = hyp_embeds.get(hyp_tok.id)
        v_r_head = ref_embeds.get(ref_head.id)
        v_r_dep  = ref_embeds.get(ref_tok.id)

    head_sim01 = to_01(cosine_similarity(v_h_head, v_r_head))
    dep_sim01  = to_01(cosine_similarity(v_h_dep,  v_r_dep))

    dep_floored = smooth_floor(dep_sim01)
    head_conf   = HEAD_BASE + HEAD_SCALE * head_sim01
    base_sim    = dep_floored * head_conf

    if base_sim == 0.0:
        return 0.0, {"head_sim": round(head_sim01,4), "dep_sim": round(dep_sim01,4),
                     "base": 0.0, "bonuses": [], "score": 0.0, "deprel_match": False}

    bonuses = []
    hyp_rel = (hyp_tok.deprel or "").lower()
    ref_rel = (ref_tok.deprel or "").lower()

    if base_sim >= BONUS_SEM_THRESHOLD:
        dr = _deprel_bonus(hyp_rel, ref_rel)
        if dr > 0:
            bonuses.append(("deprel_match", round(dr, 4)))
        if hyp_tok.upos == "VERB" and ref_tok.upos == "VERB" and head_sim01 > 0.70:
            bonuses.append(("verb_paraphrase", BONUS_VERB_PARAPHRASE))
        if hyp_rel in CORE_ARGS and ref_rel in CORE_ARGS:
            bonuses.append(("core_arg_match", BONUS_CORE_ARG_MATCH))

    if (getattr(hyp_tok, "lemma", None) and getattr(ref_tok, "lemma", None)
            and hyp_tok.lemma.lower() == ref_tok.lemma.lower()):
        bonuses.append(("exact_lemma", BONUS_EXACT_LEMMA))

    score = min(1.0, base_sim + sum(v for _, v in bonuses))

    return score, {
        "head_sim":     round(head_sim01, 4),
        "dep_sim":      round(dep_sim01,  4),
        "dep_floored":  round(dep_floored, 4),
        "head_conf":    round(head_conf,   4),
        "base":         round(base_sim,   4),
        "bonuses":      bonuses,
        "score":        round(score,      4),
        "deprel_match": hyp_rel == ref_rel,
        "hyp_rel":      hyp_rel,
        "ref_rel":      ref_rel,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Matching
# ══════════════════════════════════════════════════════════════════════════════

def greedy_matching(sim_matrix: np.ndarray):
    used = set()
    row_ind, col_ind = [], []
    for i in np.argsort(-sim_matrix.max(axis=1)):
        j = int(np.argmax(sim_matrix[i]))
        if j not in used:
            row_ind.append(i); col_ind.append(j); used.add(j)
    return row_ind, col_ind


# ══════════════════════════════════════════════════════════════════════════════
# Main scorer
# ══════════════════════════════════════════════════════════════════════════════

def compute_sentence_similarity(
    dep_h, dep_r, embedder,
    lang: str = "en", matching: str = "hungarian", debug: bool = False,
    passive_sent_penalty: float = 1.0,
    neg_sent_penalty: float = NEG_SENT_PENALTY,
    soften_structure: bool = False,
    return_details: bool = False,
    **_,
):
    """
    GRIS-DepScore for one sentence pair.

    score = W[lang] * edge_F1 + (1 - W[lang]) * sent_cos
    where W is the language-adaptive structural weight.

    Then multiplied by NEG_SENT_PENALTY (0.90) if negation mismatch,
    and by MAIN_SUBJ_PROPN_PENALTY (0.80) if PROPN subject mismatch.

    If return_details=True, returns (score, details) where `details` is a
    dict describing exactly how the score was derived: every hyp/ref edge
    pair that was matched (with the head/dep cosine similarities and
    bonuses that produced each edge's contribution), precision/recall/Fβ,
    the sentence-cosine fallback, the language blend weight, and which
    sentence-level penalties fired. See `explain_dep_score()` to pretty-print it.
    """
    # ── Token embeddings ──────────────────────────────────────────────────────
    if embedder.embedding_type == "transformer":
        hyp_embeds = embedder.embed_tokens(dep_h)
        ref_embeds = embedder.embed_tokens(dep_r)
    else:
        hyp_embeds = ref_embeds = None

    # ── Content-lemma sentence cosine ─────────────────────────────────────────
    try:
        sentence_cosine = float(to_01(
            embedder.compute_similarity(
                _content_lemma_text(dep_h),
                _content_lemma_text(dep_r),
            )
        ))
    except Exception:
        sentence_cosine = 0.5

    # ── Negation flags ────────────────────────────────────────────────────────
    h_neg = sentence_has_negation(dep_h)
    r_neg = sentence_has_negation(dep_r)
    h_passive = sentence_has_passive(dep_h)
    r_passive = sentence_has_passive(dep_r)

    # ── Edge extraction ───────────────────────────────────────────────────────
    hyp_edges = extract_edges(dep_h)
    ref_edges = extract_edges(dep_r)

    def _edge_label(edge):
        head, dep = edge
        return f"{head.text}→{dep.text} ({dep.deprel or 'dep'})"

    def _edge_struct(edge):
        head, dep = edge
        return {"head_word": head.text, "dep_word": dep.text,
                "deprel": dep.deprel or "dep"}

    matched_detail = []
    unmatched_hyp_detail, unmatched_ref_detail = [], []
    verbosity_fired = False
    length_penalty  = 1.0
    beta = 1.5

    # ── Edge F1 ───────────────────────────────────────────────────────────────
    if not hyp_edges and not ref_edges:
        edge_f1 = 1.0
        prec = rec = 1.0
    elif not hyp_edges or not ref_edges:
        edge_f1 = sentence_cosine
        prec = rec = sentence_cosine
        for e in hyp_edges:
            unmatched_hyp_detail.append({"edge": _edge_label(e), **_edge_struct(e)})
        for e in ref_edges:
            unmatched_ref_detail.append({"edge": _edge_label(e), **_edge_struct(e)})
    else:
        sim_matrix = np.zeros((len(hyp_edges), len(ref_edges)), dtype=np.float32)
        attr_matrix = [[None] * len(ref_edges) for _ in range(len(hyp_edges))]
        for i, h in enumerate(hyp_edges):
            for j, r in enumerate(ref_edges):
                sim, attr = compute_edge_similarity(
                    h, r, hyp_embeds, ref_embeds, embedder,
                    debug=debug, lang=lang,
                )
                sim_matrix[i, j] = float(sim)
                attr_matrix[i][j] = attr

        if matching == "hungarian":
            row_ind, col_ind = linear_sum_assignment(-sim_matrix)
        else:
            row_ind, col_ind = greedy_matching(sim_matrix)

        matched_i, matched_j = set(row_ind), set(col_ind)
        if return_details:
            for i, j in zip(row_ind, col_ind):
                a = attr_matrix[i][j] or {}
                matched_detail.append({
                    "hyp_edge": _edge_label(hyp_edges[i]),
                    "ref_edge": _edge_label(ref_edges[j]),
                    "hyp": _edge_struct(hyp_edges[i]),
                    "ref": _edge_struct(ref_edges[j]),
                    "similarity": round(float(sim_matrix[i, j]), 4),
                    "head_sim":   a.get("head_sim"),
                    "dep_sim":    a.get("dep_sim"),
                    "bonuses":    a.get("bonuses", []),
                    "deprel_match": a.get("deprel_match"),
                })
            for i in range(len(hyp_edges)):
                if i not in matched_i:
                    unmatched_hyp_detail.append({"edge": _edge_label(hyp_edges[i]),
                                                  **_edge_struct(hyp_edges[i])})
            for j in range(len(ref_edges)):
                if j not in matched_j:
                    unmatched_ref_detail.append({"edge": _edge_label(ref_edges[j]),
                                                  **_edge_struct(ref_edges[j])})
            matched_detail.sort(key=lambda d: d["similarity"], reverse=True)

        total_sim = float(sum(sim_matrix[i, j] for i, j in zip(row_ind, col_ind)))
        n_h, n_r  = len(hyp_edges), len(ref_edges)
        prec      = total_sim / n_h
        rec       = total_sim / n_r

        # F_beta(1.5) + verbosity penalty
        #
        # β=1.5 (recall-weighted) for all languages:
        #   WMT22 DE: ABL_C (β=1.5) = 0.3137 vs FULL (β=1.0) = 0.2687 (+0.045).
        #   MT errors are typically omissions (missing content) → recall matters more.
        #
        # Note: for zh→en, Evaluate.py now correctly passes lang="en" (parse_lang),
        # so this function always receives the TARGET TEXT language, not the source.
        # A ZH-specific β override is therefore not needed here.
        #
        # Verbosity penalty (VP): fires when hyp has >40% more edges than ref.
        #   Caps recall inflation for verbose MT systems at system level.
        #   WMT22 ZH analysis: verbosity fires on only 4% of pairs, p=0.63 vs MQM
        #   → not a quality signal for ZH, but harmless since it fires so rarely.
        VP_THRESHOLD = 1.4
        VP_FACTOR    = 0.92

        _b2  = beta ** 2   # = 2.25
        _den = _b2 * prec + rec
        f1   = (1.0 + _b2) * prec * rec / _den if _den > 0 else 0.0

        # Verbosity penalty
        if n_h > n_r * VP_THRESHOLD:
            f1 *= VP_FACTOR
            verbosity_fired = True

        lr  = min(n_h, n_r) / max(n_h, n_r)
        length_penalty = (0.85 + 0.15 * lr) if lr < 0.5 else 1.0
        edge_f1 = max(0.0, min(1.0, f1 * length_penalty))

    # ── Language-adaptive blend ───────────────────────────────────────────────
    w    = STRUCT_WEIGHT.get(lang, STRUCT_WEIGHT_DEFAULT)
    base = w * edge_f1 + (1.0 - w) * sentence_cosine
    blend_base = base

    # ── PROPN subject check ───────────────────────────────────────────────────
    base_before_propn = base
    base = _apply_main_subject_check(base, dep_h, dep_r, hyp_embeds, ref_embeds, embedder)
    propn_fired = base != base_before_propn

    # ── Negation penalty ──────────────────────────────────────────────────────
    base_before_neg = base
    neg_mismatch = h_neg != r_neg
    if neg_mismatch:
        base *= float(neg_sent_penalty)

    final_score = max(0.0, min(1.0, float(base)))

    if not return_details:
        return final_score

    details = {
        "hyp_edges": [_edge_label(e) for e in hyp_edges],
        "ref_edges": [_edge_label(e) for e in ref_edges],
        "matched":       matched_detail,
        "unmatched_hyp": unmatched_hyp_detail,
        "unmatched_ref": unmatched_ref_detail,
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "beta":      beta,
        "verbosity_penalty_fired": verbosity_fired,
        "length_penalty": round(length_penalty, 4),
        "edge_f1":   round(edge_f1, 4),
        "sentence_cosine": round(sentence_cosine, 4),
        "lang": lang,
        "blend_weight_W": w,
        "blend_score": round(blend_base, 4),
        "propn_check_fired": propn_fired,
        "score_before_propn": round(base_before_propn, 4),
        "score_before_negation": round(base_before_neg, 4),
        "negation": {
            "hyp_has_negation": h_neg, "ref_has_negation": r_neg,
            "mismatch": neg_mismatch,
            "penalty_applied": round(float(neg_sent_penalty), 4) if neg_mismatch else 1.0,
        },
        "passive": {
            "hyp_is_passive": h_passive, "ref_is_passive": r_passive,
            "mismatch": h_passive != r_passive,
            "penalty_applied": 1.0,  # WMT22 final: passive is flagged only, never penalizes
        },
        "final_score": round(final_score, 4),
    }
    return final_score, details


# ══════════════════════════════════════════════════════════════════════════════
# Human-readable explanation of a return_details=True breakdown
# ══════════════════════════════════════════════════════════════════════════════

# Plain-English names for Universal Dependencies relation labels, used to turn
# "lief→Hund (nsubj)" into "'Hund' is the subject of 'lief'".
_DEPREL_PLAIN = {
    "nsubj": "the subject of", "nsubj:pass": "the subject of",
    "csubj": "the subject of", "csubj:pass": "the subject of",
    "obj": "the object of", "iobj": "the indirect object of",
    "obl": "linked to", "obl:agent": "the agent of",
    "obl:tmod": "a time expression modifying", "obl:npmod": "a modifier of",
    "advmod": "an adverb modifying", "amod": "an adjective describing",
    "det": "a determiner for", "aux": "a helping verb for",
    "aux:pass": "a helping verb for", "cop": "a linking verb for",
    "mark": "a connector introducing", "case": "a preposition attached to",
    "nmod": "a modifier of", "nummod": "a number modifying",
    "conj": "connected to", "cc": "a conjunction linking",
    "punct": "punctuation for", "compound": "part of a compound with",
    "flat": "part of a name with", "appos": "describing",
    "xcomp": "a complement of", "ccomp": "a clausal complement of",
    "acl": "a clause modifying", "advcl": "an adverbial clause modifying",
    "root": "the main verb",
}


def _deprel_plain(rel: str) -> str:
    return _DEPREL_PLAIN.get(rel, f"related ({rel}) to")


def _quality_label(score: float) -> str:
    if score >= 0.90: return "Excellent"
    if score >= 0.75: return "Good"
    if score >= 0.60: return "Acceptable"
    if score >= 0.40: return "Weak"
    return "Poor"


def _plain_language_summary(details: dict, hyp: str = None, ref: str = None) -> list:
    """Build the human-readable narrative section (a list of lines)."""
    lines = []
    w = lines.append
    fs = details["final_score"]

    w(f"PLAIN-LANGUAGE SUMMARY")
    w("-" * 72)
    if ref is not None: w(f"Reference:  {ref}")
    if hyp  is not None: w(f"Translation: {hyp}")
    w("")
    w(f"Score: {fs:.2f} out of 1.00  →  {_quality_label(fs)} translation")
    w("")

    # What was preserved correctly
    exact, paraphrased, weak = [], [], []
    head_paraphrases = {}  # (ref_head, hyp_head) -> best head_sim seen, deduped
    for m in details["matched"]:
        hd, rd = m["hyp"]["dep_word"], m["ref"]["dep_word"]
        h_head, r_head = m["hyp"]["head_word"], m["ref"]["head_word"]
        rel = _deprel_plain(m["ref"]["deprel"])
        has_exact_lemma = any(name == "exact_lemma" for name, _ in m["bonuses"])

        if h_head.lower() != r_head.lower():
            head_paraphrases[(r_head, h_head)] = max(
                head_paraphrases.get((r_head, h_head), 0), m["head_sim"] or 0)

        if m["similarity"] >= 0.90 and has_exact_lemma:
            exact.append(f"'{rd}' — {rel} '{r_head}' — is preserved correctly")
        elif m["similarity"] >= 0.75:
            if hd.lower() == rd.lower():
                paraphrased.append(f"'{rd}' — {rel} '{r_head}' — matches")
            else:
                paraphrased.append(
                    f"'{rd}' was translated as '{hd}' — a close paraphrase, "
                    f"still {rel} the equivalent of '{r_head}'")
        else:
            weak.append(f"'{rd}' (reference) vs '{hd}' (translation) — "
                         f"{rel} '{r_head}', but the match is weak")

    if exact:
        w("What's preserved exactly:")
        for e in exact: w(f"  ✅ {e}")
        w("")
    if head_paraphrases:
        w("Word choice differences (same grammatical role, different word):")
        for (r_head, h_head), sim in head_paraphrases.items():
            closeness = ("a close synonym" if sim >= 0.75 else
                         "a related word" if sim >= 0.5 else "a different word")
            w(f"  🔤 Reference says '{r_head}', translation says '{h_head}' — "
              f"{closeness} (similarity={sim:.2f})")
        w("")
    if paraphrased:
        w("What's preserved with minor wording differences:")
        for p in paraphrased: w(f"  🟡 {p}")
        w("")
    if weak:
        w("Parts that differ more substantially:")
        for x in weak: w(f"  🟠 {x}")
        w("")

    if details["unmatched_ref"]:
        w("Missing from the translation (present in the reference but not the hypothesis):")
        for e in details["unmatched_ref"]:
            w(f"  ❌ '{e['dep_word']}' ({_deprel_plain(e['deprel'])} '{e['head_word']}')")
        w("")
    if details["unmatched_hyp"]:
        w("Extra content in the translation (not present in the reference):")
        for e in details["unmatched_hyp"]:
            w(f"  ➕ '{e['dep_word']}' ({_deprel_plain(e['deprel'])} '{e['head_word']}')")
        w("")

    # Grammar-level issues
    issues = []
    n, p = details["negation"], details["passive"]
    if n["mismatch"]:
        issues.append(
            "⚠️  Negation mismatch — one sentence says 'not' (or similar) and the "
            "other doesn't. This changes the actual meaning, so it lowers the score.")
    if p["mismatch"]:
        issues.append(
            "ℹ️  Voice differs (active vs. passive) between the two sentences. "
            "This is noted but does NOT lower the score — it's considered a valid "
            "paraphrase style.")
    if details["propn_check_fired"]:
        issues.append(
            "⚠️  The main named-entity subject doesn't match between the translation "
            "and the reference (e.g. a different name/entity as the subject).")
    if details["verbosity_penalty_fired"]:
        issues.append(
            "⚠️  The translation is noticeably longer/more verbose than the reference.")
    if issues:
        w("Grammar-level checks:")
        for i in issues: w(f"  {i}")
        w("")

    # One-line "why this score" takeaway
    w("Why this score:")
    w(f"  The grammatical structure matched {details['edge_f1']*100:.0f}% "
      f"(precision={details['precision']:.2f}, recall={details['recall']:.2f}), "
      f"and overall meaning similarity was {details['sentence_cosine']*100:.0f}%. "
      f"For this language, structure counts for {details['blend_weight_W']*100:.0f}% "
      f"of the score and overall meaning for the remaining "
      f"{(1-details['blend_weight_W'])*100:.0f}%.")
    if n["mismatch"]:
        w(f"  The negation mismatch above reduced the score by "
          f"{(1-n['penalty_applied'])*100:.0f}%.")

    return lines


def explain_dep_score(details: dict, hyp: str = None, ref: str = None,
                       top_n: int = 10, technical: bool = True) -> str:
    """
    Render a `details` dict from compute_sentence_similarity(..., return_details=True)
    (or the details list returned by scorer.compute_DepScore_emb(..., return_details=True))
    as a human-readable explanation of how the GRIS-DepScore was derived.

    Always prints a plain-language summary first (what was preserved, what's
    missing/added, and why the score came out where it did — no jargon).

    If technical=True (default), also prints the full step-by-step numeric
    trace (edge matching, precision/recall/Fβ, blend formula, penalties) below
    the plain-language summary, for readers who want the exact mechanics.
    Pass technical=False to show only the plain-language summary.

    Prints the explanation and also returns it as a string.
    """
    lines = _plain_language_summary(details, hyp=hyp, ref=ref)

    if not technical:
        text = "\n".join(lines)
        print(text)
        return text

    lines.append("")
    lines.append("=" * 72)
    lines.append("TECHNICAL BREAKDOWN")
    w = lines.append
    w("=" * 72)
    if hyp is not None or ref is not None:
        if ref is not None: w(f"REF: {ref}")
        if hyp is not None: w(f"HYP: {hyp}")
        w("-" * 72)

    w(f"GRIS-DepScore = {details['final_score']}")
    w("=" * 72)

    w(f"\n1. Dependency edges extracted")
    w(f"   hyp: {len(details['hyp_edges'])} edges | ref: {len(details['ref_edges'])} edges")

    w(f"\n2. Edge matching (Hungarian assignment) — top {top_n} by similarity")
    for m in details["matched"][:top_n]:
        bonus_str = (", bonuses: " + ", ".join(f"{n}(+{v})" for n, v in m["bonuses"])
                     if m["bonuses"] else "")
        w(f"   {m['hyp_edge']:<28} ↔ {m['ref_edge']:<28} "
          f"sim={m['similarity']:.3f} (head={m['head_sim']}, dep={m['dep_sim']}{bonus_str})")
    n_more = len(details["matched"]) - top_n
    if n_more > 0:
        w(f"   ... and {n_more} more matched edge pair(s)")
    if details["unmatched_hyp"]:
        w(f"   unmatched HYP edges (no ref counterpart): "
          f"{', '.join(e['edge'] for e in details['unmatched_hyp'])}")
    if details["unmatched_ref"]:
        w(f"   unmatched REF edges (missed by hyp):      "
          f"{', '.join(e['edge'] for e in details['unmatched_ref'])}")

    w(f"\n3. Precision / Recall / Fβ(β={details['beta']})")
    w(f"   precision={details['precision']}, recall={details['recall']}"
      f" → edge_F1={details['edge_f1']}")
    if details["verbosity_penalty_fired"]:
        w(f"   verbosity penalty applied (hyp has >40% more edges than ref)")
    if details["length_penalty"] != 1.0:
        w(f"   length penalty applied: ×{details['length_penalty']}"
          f" (hyp/ref edge counts differ substantially)")

    w(f"\n4. Language-adaptive blend (lang='{details['lang']}', W={details['blend_weight_W']})")
    w(f"   blend = W×edge_F1 + (1-W)×sentence_cosine")
    w(f"         = {details['blend_weight_W']}×{details['edge_f1']} + "
      f"{round(1 - details['blend_weight_W'], 4)}×{details['sentence_cosine']}"
      f" = {details['blend_score']}")

    w(f"\n5. Sentence-level checks")
    p = details["passive"]
    flag = "MISMATCH (flagged only — never penalizes, WMT22 final)" if p["mismatch"] else "match"
    w(f"   passive voice: hyp={p['hyp_is_passive']}, ref={p['ref_is_passive']} → {flag}")
    n = details["negation"]
    if n["mismatch"]:
        w(f"   negation: hyp={n['hyp_has_negation']}, ref={n['ref_has_negation']} → "
          f"MISMATCH → ×{n['penalty_applied']} penalty applied")
    else:
        w(f"   negation: hyp={n['hyp_has_negation']}, ref={n['ref_has_negation']} → match, no penalty")
    if details["propn_check_fired"]:
        w(f"   named-entity subject check: fired → "
          f"{details['score_before_propn']} → ×0.80 → {details['score_before_negation']}")

    w(f"\n6. Final score")
    w(f"   {details['blend_score']} "
      f"{'× ' + str(n['penalty_applied']) + ' (negation) ' if n['mismatch'] else ''}"
      f"{'× 0.80 (named-entity subject) ' if details['propn_check_fired'] else ''}"
      f"= {details['final_score']}")
    w("=" * 72)

    text = "\n".join(lines)
    print(text)
    return text


def compute_edge_similarity_v23(
    hyp_edge, ref_edge, hyp_embeds, ref_embeds, embedder,
    debug: bool = False, soften_structure: bool = False,
    lang: str = "en", **_,
):
    score, attr = compute_edge_similarity(
        hyp_edge, ref_edge, hyp_embeds, ref_embeds, embedder,
        debug=debug, lang=lang,
    )
    return score, attr.get("bonuses", [])