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
    soften_structure: bool = False, **_,
):
    """
    GRIS-DepScore for one sentence pair.

    score = W[lang] * edge_F1 + (1 - W[lang]) * sent_cos
    where W is the language-adaptive structural weight.

    Then multiplied by NEG_SENT_PENALTY (0.90) if negation mismatch,
    and by MAIN_SUBJ_PROPN_PENALTY (0.80) if PROPN subject mismatch.
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

    # ── Edge extraction ───────────────────────────────────────────────────────
    hyp_edges = extract_edges(dep_h)
    ref_edges = extract_edges(dep_r)

    # ── Edge F1 ───────────────────────────────────────────────────────────────
    if not hyp_edges and not ref_edges:
        edge_f1 = 1.0
    elif not hyp_edges or not ref_edges:
        edge_f1 = sentence_cosine
    else:
        sim_matrix = np.zeros((len(hyp_edges), len(ref_edges)), dtype=np.float32)
        for i, h in enumerate(hyp_edges):
            for j, r in enumerate(ref_edges):
                sim, _ = compute_edge_similarity(
                    h, r, hyp_embeds, ref_embeds, embedder,
                    debug=debug, lang=lang,
                )
                sim_matrix[i, j] = float(sim)

        if matching == "hungarian":
            row_ind, col_ind = linear_sum_assignment(-sim_matrix)
        else:
            row_ind, col_ind = greedy_matching(sim_matrix)

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

        _b2  = 1.5 ** 2   # = 2.25
        _den = _b2 * prec + rec
        f1   = (1.0 + _b2) * prec * rec / _den if _den > 0 else 0.0

        # Verbosity penalty
        if n_h > n_r * VP_THRESHOLD:
            f1 *= VP_FACTOR

        lr  = min(n_h, n_r) / max(n_h, n_r)
        lp  = (0.85 + 0.15 * lr) if lr < 0.5 else 1.0
        edge_f1 = max(0.0, min(1.0, f1 * lp))

    # ── Language-adaptive blend ───────────────────────────────────────────────
    w    = STRUCT_WEIGHT.get(lang, STRUCT_WEIGHT_DEFAULT)
    base = w * edge_f1 + (1.0 - w) * sentence_cosine

    # ── PROPN subject check ───────────────────────────────────────────────────
    base = _apply_main_subject_check(base, dep_h, dep_r, hyp_embeds, ref_embeds, embedder)

    # ── Negation penalty ──────────────────────────────────────────────────────
    if h_neg != r_neg:
        base *= float(neg_sent_penalty)

    return max(0.0, min(1.0, float(base)))


# ── Back-compat alias ─────────────────────────────────────────────────────────
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