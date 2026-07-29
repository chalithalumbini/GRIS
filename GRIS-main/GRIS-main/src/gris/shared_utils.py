"""
shared_utils.py - Shared utilities for both DepScore and GRIS-SynGram

CHANGES v2 (beat-BERTScore tuning):
- PEN_STRUCTURE:  0.40 -> 0.15  WMT22 ablation shows structure penalty hurts DE→EN
                                 (GRIS-DepScore_NO_STRUCTURE scores 0.2874 vs FULL 0.2687).
                                 German topicalisation/scrambling means valid translations
                                 legitimately change nsubj/obl roles; penalising this is a
                                 false positive for paraphrastic translations.
- SEM_OVERRIDE:   0.80 -> 0.65  Cosine ≥0.65 in multilingual MiniLM space already indicates
                                 near-synonymy; reduce threshold so strong embeddings suppress
                                 remaining structure penalty more aggressively.
- PEN_MORPHOLOGY: unchanged 0.30
- NEG_EDGE_PENALTY: unchanged 0.25
- Added depth_gated_unmatched_weight() helper for SynGram coverage penalty fix.
- Removed BONUS_PARAPHRASE_VERB from final formula (still exported for ngram_matcher)
"""

from typing import Set, List, Tuple, Any
from collections import defaultdict

# =========================
# SHARED CONSTANTS
# =========================
EMB_FLOOR = 0.15   # lowered: German MiniLM embeddings for paraphrases often score 0.22-0.28; 0.20 floor was too aggressive

# Structure penalty reduced: WMT22 ablation shows NO_STRUCTURE outperforms FULL for DE→EN
# (0.2874 vs 0.2687 Spearman). German topicalisation/scrambling means valid paraphrastic
# translations legitimately shift nsubj↔obl roles; 0.40 was a false-positive factory.
PEN_STRUCTURE    = 0.15   # was 0.40
PEN_MORPHOLOGY   = 0.30
NEG_EDGE_PENALTY = 0.25
COORD_PENALTY    = 0.05
ENTITY_PENALTY   = 0.35

BONUS_EXACT_MATCH     = 0.10
BONUS_PARAPHRASE_VERB = 0.05

SEM_OVERRIDE       = 0.65   # was 0.80: cosine ≥0.65 in MiniLM space already indicates near-synonymy
SEM_OVERRIDE_SCALE = 0.30

NEG_SENT_PENALTY     = 0.70
TENSE_SENT_PENALTY   = 0.75   # root verb tense mismatch (e.g. past vs present)

VOICE_COMPATIBLE = {
    frozenset({"nsubj", "obl:agent"}),
    frozenset({"nsubj", "obl"}),
    frozenset({"obj", "nsubj:pass"}),
    frozenset({"obj", "nsubj"}),
    frozenset({"ARG", "ARG"}),
}

SOFT_COMPATIBLE = {
    frozenset({"nsubj", "csubj"}),
    frozenset({"obj", "obl"}),
    frozenset({"obl", "nmod"}),
    frozenset({"amod", "advmod"}),
    frozenset({"aux", "cop"}),
    frozenset({"conj", "conj"}),
    frozenset({"ARG", "nmod"}),
}

CORE_ARGS = {"nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp", "nsubj:pass", "ARG"}
STOP_UPOS = {"DET", "AUX", "PART", "SCONJ", "CCONJ", "INTJ", "PUNCT"}


# =========================
# SIMILARITY HELPERS
# =========================
def smooth_floor(sim01: float) -> float:
    """Apply floor to similarity already in [0,1]."""
    if sim01 <= EMB_FLOOR:
        return 0.0
    return (sim01 - EMB_FLOOR) / (1.0 - EMB_FLOOR)


def to_01(sim_m11: float) -> float:
    """Map [-1,1] to [0,1]. Used by DepScore before smooth_floor."""
    return max(0.0, min(1.0, (sim_m11 + 1.0) / 2.0))


# =========================
# NEGATION DETECTION
# =========================
def _neg_lexicon() -> Set[str]:
    return {
        "nicht", "nie", "kein", "keine", "keinen", "keinem", "keiner", "keines",
        "nichts", "weder", "ohne",
        "not", "no", "never", "nobody", "nothing", "nowhere",
        "ei", "en",
        "ne", "pas", "non",
        "no", "nunca", "nada", "nadie",
    }


def has_negation_local(tok, head) -> bool:
    """LOCAL negation: explicit neg dep or negation lemmas."""
    if (getattr(tok, "deprel", None) or "").lower() == "neg":
        return True
    neg_lemmas = _neg_lexicon()
    for t in (tok, head):
        lem = (getattr(t, "lemma", None) or "").lower().strip()
        txt = (getattr(t, "text",  None) or "").lower().strip()
        if (lem and lem in neg_lemmas) or (txt and txt in neg_lemmas):
            return True
    return False


def sentence_has_negation(tokens) -> bool:
    """GLOBAL negation: sentence-level detection."""
    neg_lemmas = _neg_lexicon()
    for t in tokens:
        rel = (getattr(t, "deprel", "") or "").lower()
        if rel == "neg":
            return True
        lem = (getattr(t, "lemma", None) or "").lower().strip()
        txt = (getattr(t, "text",  None) or "").lower().strip()
        if (lem and lem in neg_lemmas) or (txt and txt in neg_lemmas):
            return True
    return False


# =========================
# VOICE DETECTION
# =========================
def _is_passive_head(head_tok) -> bool:
    feats = getattr(head_tok, "feats", None) or {}
    return bool(isinstance(feats, dict) and feats.get("Voice") == "Pass")


def sentence_has_passive(tokens) -> bool:
    """GLOBAL passive detection."""
    id2 = {t.id: t for t in tokens}
    children = defaultdict(list)
    for t in tokens:
        hid = getattr(t, "head", 0) or 0
        if hid and hid != 0:
            children[hid].append(t.id)

    for hid, deps in children.items():
        head = id2.get(hid)
        if head is None:
            continue
        up = (getattr(head, "upos", "") or "X").upper()
        if up not in ("VERB", "AUX"):
            continue
        if _is_passive_head(head):
            return True
        for did in deps:
            rel0 = (getattr(id2[did], "deprel", "") or "").lower()
            if rel0 in ("aux:pass", "nsubj:pass", "csubj:pass"):
                return True

    return False


def is_voice_compatible(rel1: str, rel2: str) -> bool:
    return frozenset({rel1, rel2}) in VOICE_COMPATIBLE


def is_soft_compatible(rel1: str, rel2: str) -> bool:
    return frozenset({rel1, rel2}) in SOFT_COMPATIBLE


# =========================
# MORPHOLOGY CHECKING
# =========================
def check_morphology_penalties(
    hyp_tok,
    ref_tok,
    penalties: List[Tuple[str, float]]
) -> None:
    """Check morphological features and append penalties in-place."""
    if hyp_tok is None or ref_tok is None:
        return
    # Extended feature set: Tense + VerbForm + Aspect catch tense changes
    # including progressive ("I am going" VerbForm=Part) vs simple past ("went" Tense=Past)
    for feat in {"Number", "Tense", "Polarity", "VerbForm", "Aspect"}:
        hv = (getattr(hyp_tok, "feats", None) or {}).get(feat)
        rv = (getattr(ref_tok, "feats", None) or {}).get(feat)
        if hv and rv and hv != rv:
            weight = 1.5 if feat == "Polarity" else 1.0
            penalties.append((f"morph_{feat}", PEN_MORPHOLOGY * weight))


def _get_root_verb(tokens) -> Any:
    """Return the root token if it is a VERB, else return any VERB child of root."""
    id2 = {t.id: t for t in tokens}
    # First pass: find explicit root
    for t in tokens:
        if getattr(t, "head", None) == 0:
            if (getattr(t, "upos", "") or "").upper() in ("VERB", "AUX"):
                return t
    # Second pass: find VERB whose head is the root
    root_id = next((t.id for t in tokens if getattr(t, "head", None) == 0), None)
    if root_id:
        for t in tokens:
            if getattr(t, "head", None) == root_id:
                if (getattr(t, "upos", "") or "").upper() in ("VERB", "AUX"):
                    return t
    return None


def _get_effective_tense(tokens) -> str:
    """
    Get the effective tense of a sentence by examining the root verb
    and its AUX children.

    Only returns the Tense feature — NOT VerbForm or Aspect.
    This prevents false mismatches between active and passive constructions
    where VerbForm differs (Fin vs Part) but tense is the same.

    Examples:
      'The dog bit the man'         → root='bit'    Tense=Past → 'Past'
      'The man was bitten by...'    → root='bitten' no Tense, AUX 'was' Tense=Past → 'Past'
      'I am going'                  → root='going'  no Tense, AUX 'am' Tense=Pres → 'Pres'
      'I went'                      → root='went'   Tense=Past → 'Past'
    """
    root = _get_root_verb(tokens)
    if root is None:
        return ""

    root_feats = getattr(root, "feats", None) or {}
    tense      = root_feats.get("Tense", "")

    # If root verb has no Tense (e.g. passive participle), look at AUX children
    if not tense:
        for t in tokens:
            if getattr(t, "head", None) == root.id:
                if (getattr(t, "upos", "") or "").upper() == "AUX":
                    aux_feats = getattr(t, "feats", None) or {}
                    aux_tense = aux_feats.get("Tense", "")
                    if aux_tense:
                        tense = aux_tense
                        break

    return tense


def sentence_has_tense_mismatch(hyp_tokens, ref_tokens) -> bool:
    """
    GLOBAL tense mismatch detection.

    Returns True if the effective tense of the hypothesis root verb
    differs from the reference root verb.

    Examples that return True:
      'I am going' (Pres:Part:Prog) vs 'I went' (Past:Fin)
      'He runs'    (Pres:Fin)       vs 'He ran'  (Past:Fin)
      'She will go'(Fut)            vs 'She went' (Past:Fin)

    Examples that return False:
      'I am going' vs 'I am travelling'  (both Pres:Part:Prog)
      'He ran fast' vs 'He ran slowly'   (both Past:Fin)
    """
    hyp_tense = _get_effective_tense(hyp_tokens)
    ref_tense = _get_effective_tense(ref_tokens)
    if not hyp_tense or not ref_tense:
        return False
    return hyp_tense != ref_tense


# =========================
# STRUCTURAL PENALTIES
# =========================
def compute_structure_penalty(
    hyp_rel: str,
    ref_rel: str,
    sem_strong: bool
) -> Tuple[str, float]:
    """Compute structural penalty for a single relation mismatch."""
    if hyp_rel == "conj" and ref_rel == "conj":
        return ("coord_soft", COORD_PENALTY)

    if hyp_rel == ref_rel:
        return ("exact_match", 0.0)

    if is_voice_compatible(ref_rel, hyp_rel):
        pen = PEN_STRUCTURE * 0.10
        if sem_strong:
            pen *= SEM_OVERRIDE_SCALE
        return ("voice_compatible_local", pen)

    rel_set = frozenset({hyp_rel, ref_rel})
    if rel_set in SOFT_COMPATIBLE:
        pen = PEN_STRUCTURE * 0.25
        if sem_strong:
            pen *= SEM_OVERRIDE_SCALE
        return ("structure_soft", pen)

    if hyp_rel in CORE_ARGS or ref_rel in CORE_ARGS:
        pen = PEN_STRUCTURE
        if sem_strong:
            pen *= SEM_OVERRIDE_SCALE
        return ("structure_core", pen)

    pen = PEN_STRUCTURE * 0.5
    if sem_strong:
        pen *= SEM_OVERRIDE_SCALE
    return ("structure_other", pen)


def compute_structure_penalty_multi(
    hyp_rels: List[str],
    ref_rels: List[str],
    sem_strong: bool
) -> List[Tuple[str, float]]:
    """Compute structural penalties for n-gram relation sets."""
    penalties = []

    if set(hyp_rels) == set(ref_rels):
        return penalties

    if all(r == "conj" for r in hyp_rels + ref_rels):
        penalties.append(("coord_soft", COORD_PENALTY))
        return penalties

    hyp_set = set(hyp_rels)
    ref_set = set(ref_rels)

    if any(is_voice_compatible(h, r) for h in hyp_set for r in ref_set):
        pen = PEN_STRUCTURE * 0.10
        if sem_strong:
            pen *= SEM_OVERRIDE_SCALE
        penalties.append(("voice_compatible_local", pen))
        return penalties

    if any(is_soft_compatible(h, r) for h in hyp_set for r in ref_set):
        pen = PEN_STRUCTURE * 0.25
        if sem_strong:
            pen *= SEM_OVERRIDE_SCALE
        penalties.append(("structure_soft", pen))
        return penalties

    has_core = any(r in CORE_ARGS for r in hyp_rels + ref_rels)
    if has_core:
        pen = PEN_STRUCTURE
        if sem_strong:
            pen *= SEM_OVERRIDE_SCALE
        penalties.append(("structure_core", pen))
    else:
        pen = PEN_STRUCTURE * 0.5
        if sem_strong:
            pen *= SEM_OVERRIDE_SCALE
        penalties.append(("structure_other", pen))

    return penalties


# =========================
# BONUS COMPUTATION
# =========================
def compute_bonuses(hyp_tok, ref_tok, head_sim: float) -> float:
    bonus = 0.0
    hyp_lem = getattr(hyp_tok, "lemma", None)
    ref_lem = getattr(ref_tok, "lemma", None)
    if hyp_lem and ref_lem and hyp_lem.lower() == ref_lem.lower():
        bonus += BONUS_EXACT_MATCH
    hyp_pos = getattr(hyp_tok, "upos", None)
    ref_pos = getattr(ref_tok, "upos", None)
    if hyp_pos == "VERB" and ref_pos == "VERB" and to_01(head_sim) > 0.7:
        bonus += BONUS_PARAPHRASE_VERB
    return bonus


# =========================
# ENTITY / NOMINALIZATION
# =========================
def check_entity_mismatch(
    hyp_tok, ref_tok,
    head_sim: float, dep_sim: float,
    penalties: List[Tuple[str, float]]
) -> None:
    hyp_pos = getattr(hyp_tok, "upos", None)
    ref_pos = getattr(ref_tok, "upos", None)
    if hyp_pos in {"NOUN", "PROPN"} and ref_pos in {"NOUN", "PROPN"}:
        hyp_rel = getattr(hyp_tok, "deprel", None)
        ref_rel = getattr(ref_tok, "deprel", None)
        if to_01(head_sim) > 0.6 and hyp_rel == ref_rel and to_01(dep_sim) < 0.15:
            penalties.append(("entity_mismatch", ENTITY_PENALTY))


def check_nominalization(
    hyp_tok, ref_tok,
    head_sim: float,
    penalties: List[Tuple[str, float]]
) -> None:
    hyp_pos = getattr(hyp_tok, "upos", None)
    ref_pos = getattr(ref_tok, "upos", None)
    if ((hyp_pos == "NOUN" and ref_pos == "VERB") or
            (hyp_pos == "VERB" and ref_pos == "NOUN")):
        if to_01(head_sim) > 0.7:
            penalties.append(("nominalization", 0.1))

# =========================
# ROLE SWAP DETECTION
# =========================
def _role_root_verb(dep_parse):
    for t in dep_parse.tokens:
        if getattr(t, "head", 0) == 0:
            if (getattr(t, "upos", "") or "X").upper() in ("VERB", "AUX"):
                return t
    return None

def _role_children_of(dep_parse, head_id):
    return [t for t in dep_parse.tokens if getattr(t, "head", None) == head_id]

def _role_subj_obj_lemmas(dep_parse):
    root = _role_root_verb(dep_parse)
    if root is None:
        return None, None
    subj = obj = None
    for ch in _role_children_of(dep_parse, root.id):
        rel   = (getattr(ch, "deprel", "") or "").lower()
        lemma = (getattr(ch, "lemma", None) or getattr(ch, "text", "") or "").lower().strip()
        if not lemma:
            continue
        if rel.startswith("nsubj") or rel.startswith("csubj"):
            subj = lemma
        elif rel in ("obj", "iobj"):
            obj = lemma
    return subj, obj

def role_swap_flag(dep_h, dep_r, st_model, margin=0.10):
    """
    Returns True if subject and object appear to be swapped between
    hypothesis and reference. Uses embedding similarity to compare
    aligned (hyp_subj↔ref_subj, hyp_obj↔ref_obj) vs crossed
    (hyp_subj↔ref_obj, hyp_obj↔ref_subj) similarity.
    """
    import numpy as np
    h_subj, h_obj = _role_subj_obj_lemmas(dep_h)
    r_subj, r_obj = _role_subj_obj_lemmas(dep_r)
    if not (h_subj and h_obj and r_subj and r_obj):
        return False
    vecs = st_model.encode(
        [h_subj, h_obj, r_subj, r_obj],
        convert_to_numpy=True, normalize_embeddings=False,
    )
    def _cos(u, v):
        d = np.linalg.norm(u) * np.linalg.norm(v)
        return float(np.dot(u, v) / d) if d > 0 else 0.0
    hs, ho, rs, ro = vecs
    aligned = 0.5 * ((_cos(hs, rs) + 1.0) / 2.0 + (_cos(ho, ro) + 1.0) / 2.0)
    crossed = 0.5 * ((_cos(hs, ro) + 1.0) / 2.0 + (_cos(ho, rs) + 1.0) / 2.0)
    return crossed > aligned + margin


# =========================
# SYNGRAM COVERAGE PENALTY HELPER
# =========================

def depth_gated_unmatched_weight(
    ref_id: int,
    ref_importance: float,
    depth_map: dict,
    shallow_depth: int = 2,
    deep_discount: float = 0.20,
) -> float:
    """
    Return the coverage-penalty weight for an unmatched reference head.

    Motivation
    ----------
    For German→English MT, the hypothesis often uses fewer explicit syntactic
    heads than the reference (German head-final structures, compound verbs,
    elided pronouns).  Adding ALL unmatched ref heads equally to the
    denominator systematically penalises valid translations.

    Fix: deep heads (depth ≥ shallow_depth) contribute only deep_discount
    fraction of their importance to the denominator, because:
      - Depth ≥ 2 heads are predominantly determiners' heads, nominal
        modifiers, or prepositional objects — peripheral, not core.
      - Missing them rarely signals a translation error; it usually reflects
        structural divergence between German and English clause shapes.

    Parameters
    ----------
    ref_id        : token id of the unmatched ref head
    ref_importance: harmonic depth-weight  1/(1+depth) from _build_importance
    depth_map     : token_id -> depth dict from _build_depth_map
    shallow_depth : depth threshold below which full importance is used
    deep_discount : fraction of importance used for deep unmatched heads

    Returns
    -------
    float : effective penalty weight to add to denominator
    """
    depth = depth_map.get(ref_id, 3)
    if depth < shallow_depth:
        return ref_importance          # shallow/root heads: full penalty
    return ref_importance * deep_discount  # deep heads: discounted penalty