"""
ngram_matcher.py - GRIS-SynGram matcher  (v8)

CHANGES v8 (beat-BERTScore tuning)
------------------------------------
- LanguageConfig thresholds lowered (de: 0.63→0.55, en: 0.65→0.58, others similarly).
  Motivation: multilingual MiniLM cosine for valid DE→EN synonym pairs (e.g.
  "Fahrzeug"→"vehicle" vs "car") clusters around 0.58–0.62. The v7 threshold
  was gating out legitimate paraphrase matches.
- _adaptive_threshold floor lowered 0.40→0.35 so n=3/4 orders can still match
  when the (now lower) base threshold minus order reductions would have hit the floor.
- Order weights moved to SynGramConfig in ngram_extractor.py:
  [0.50, 0.25, 0.15, 0.10] → [0.25, 0.40, 0.25, 0.10]
  Rationale: n=1 unigrams are token-level cosine similarity — functionally
  equivalent to BERTScore recall without its contextual advantage. n=2 bigrams
  encode predicate-argument structure (head+dep pairs), which is where GRIS-SynGram
  has a genuine advantage. Shifting weight to bigrams forces MQM-correlated
  syntactic structure to dominate scoring.

CHANGES v7 (kept for reference)
----------------------------------
- Simplified base_sim: smooth_floor removed; raw head_cos01 used directly.
- Bonus applied AFTER penalty multiplication.
- Recall denominator fixed to standard len(ref_ngrams).
- Pairwise n-gram similarity caching added (_SIM_CACHE).
- SynGramConfig imported from ngram_extractor.
"""

import re
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from gris.ngram_extractor import SynGramConfig, DEFAULT_CONFIG

NGram     = Tuple[str, List[str], List[Any]]
EmbedDict = Dict[int, np.ndarray]

# ---------------------------------------------------------------------------
# Per-sentence-pair similarity cache
# ---------------------------------------------------------------------------
_SIM_CACHE: Dict[Tuple, float] = {}


def clear_sim_cache() -> None:
    """Call once per sentence pair before scoring begins."""
    _SIM_CACHE.clear()


def _cache_key(hyp_ngram: NGram, ref_ngram: NGram) -> Tuple:
    hp, hl, _ = hyp_ngram
    rp, rl, _ = ref_ngram
    return (hp, tuple(hl), rp, tuple(rl))


# ---------------------------------------------------------------------------
# Language threshold table
# ---------------------------------------------------------------------------

class LanguageConfig:
    # Thresholds lowered to recover valid paraphrase matches that multilingual
    # MiniLM scores in the 0.55–0.62 range (e.g. near-synonymous German→EN pairs).
    # The adaptive per-order reduction (-0.02 per order) still applies on top.
    RECOMMENDED_THRESHOLDS = {
        "en": 0.58, "de": 0.55, "fr": 0.55,
        "es": 0.55, "fi": 0.52, "zh": 0.50,
    }

    @staticmethod
    def get_recommended_threshold(lang: str) -> float:
        base = (lang or "en").split("_")[0].lower()
        return float(LanguageConfig.RECOMMENDED_THRESHOLDS.get(base, 0.58))


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------

def _adaptive_threshold(base: float, order: int) -> float:
    # Per-order reductions on top of the (now lower) base threshold.
    # Floor lowered from 0.40 to 0.35 so deep n-gram orders can still match
    # when the base threshold is already 0.55 and order reductions accumulate.
    adjust_map = {1: -0.02, 2: -0.04, 3: -0.06, 4: -0.08}
    return max(0.35, min(0.85, base + adjust_map.get(order, -0.08)))


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def _cos01(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    c = float(np.dot(a, b) / denom)
    return max(0.0, min(1.0, (c + 1.0) / 2.0))


def _get_token_vec(toks, embed_dict, embedder, lemmas, idx=0):
    if embed_dict and toks and idx < len(toks):
        tid = getattr(toks[idx], "id", None)
        if tid is not None and tid in embed_dict:
            return embed_dict[tid]
    if lemmas and idx < len(lemmas):
        raw = embedder.encode(lemmas[idx])
        return np.asarray(raw, dtype=np.float32)
    return None


# ---------------------------------------------------------------------------
# Pattern helpers
# ---------------------------------------------------------------------------

def _count_args_in_pattern(pattern: str) -> int:
    m = re.search(r"\[(.*?)\]", pattern)
    if not m:
        return 0
    return sum(
        1 for part in m.group(1).split(",")
        if part.strip().startswith("ARG")
    )


def _structure_penalty_syngram(hyp_pat: str, ref_pat: str,
                                 cfg: SynGramConfig) -> float:
    """
    Fires only on ARG count mismatch — argument role reversal.
    All other relation differences are handled by embedding similarity.
    """
    if hyp_pat == ref_pat:
        return 0.0
    if _count_args_in_pattern(hyp_pat) != _count_args_in_pattern(ref_pat):
        return cfg.arg_count_mismatch_penalty
    return 0.0


# ---------------------------------------------------------------------------
# Core similarity
# ---------------------------------------------------------------------------

def compute_ngram_similarity(
    hyp_ngram:  NGram,
    ref_ngram:  NGram,
    embedder,
    cfg:        SynGramConfig       = DEFAULT_CONFIG,
    hyp_embeds: Optional[EmbedDict]  = None,
    ref_embeds: Optional[EmbedDict]  = None,
) -> float:
    """
    Similarity between two STAR n-grams (v7 simplified formula).

        base_sim      = head_cos01
        penalised_sim = base_sim * (1 - total_penalty)
        final_sim     = min(1.0, penalised_sim + bonus)

    Penalties:
        arg_count_mismatch_penalty   if ARG slot counts differ
        neg_edge_penalty             if neg:PART in one pattern only

    Bonus (applied after penalty, not scaled by it):
        exact_match_bonus            if same lemma, UPOS in exact_match_upos
    """
    key = _cache_key(hyp_ngram, ref_ngram)
    if key in _SIM_CACHE:
        return _SIM_CACHE[key]

    hyp_pat, hyp_lem, hyp_toks = hyp_ngram
    ref_pat, ref_lem, ref_toks = ref_ngram

    if (not hyp_lem and not hyp_toks) or (not ref_lem and not ref_toks):
        result = 0.5 if hyp_pat == ref_pat else 0.0
        _SIM_CACHE[key] = result
        return result

    hH = _get_token_vec(hyp_toks, hyp_embeds, embedder, hyp_lem, 0)
    rH = _get_token_vec(ref_toks, ref_embeds, embedder, ref_lem, 0)
    if hH is None or rH is None:
        result = 0.5 if hyp_pat == ref_pat else 0.0
        _SIM_CACHE[key] = result
        return result

    # v7: raw cosine similarity, no smooth_floor
    base_sim = _cos01(hH, rH)

    # Penalties
    total_penalty = _structure_penalty_syngram(hyp_pat, ref_pat, cfg)
    if ("neg:PART" in hyp_pat) != ("neg:PART" in ref_pat):
        total_penalty += cfg.neg_edge_penalty
    total_penalty = min(total_penalty, 0.85)

    penalised_sim = base_sim * (1.0 - total_penalty)

    # Bonus applied AFTER penalisation
    bonus = 0.0
    hl = hyp_lem[0].lower() if hyp_lem else ""
    rl = ref_lem[0].lower() if ref_lem else ""
    hyp_upos = getattr(hyp_toks[0], "upos", None) if hyp_toks else None
    if hl and hl == rl and hyp_upos in cfg.exact_match_upos:
        bonus = cfg.exact_match_bonus

    result = max(0.0, min(1.0, float(penalised_sim + bonus)))
    _SIM_CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# Per-order F1
# ---------------------------------------------------------------------------

def compute_precision_for_n(
    hyp_ngrams,
    ref_ngrams,
    embedder,
    matching:             str                  = "greedy",
    similarity_threshold: float                = 0.63,
    adaptive_threshold:   bool                 = True,
    order:                int                  = 1,
    hyp_embeds:           Optional[EmbedDict]  = None,
    ref_embeds:           Optional[EmbedDict]  = None,
    cfg:                  SynGramConfig        = DEFAULT_CONFIG,
) -> Tuple[float, int, int]:
    """
    F1 for a single n-gram order.

    Precision = sum(matched_sim) / |hyp|
    Recall    = sum(matched_sim) / |ref|      (v7: always ref count)
    F1        = 2PR / (P + R)
    """
    if not hyp_ngrams or not ref_ngrams:
        return 0.0, 0, 0

    thr = (_adaptive_threshold(similarity_threshold, order)
           if adaptive_threshold else similarity_threshold)

    def _sim(h, r):
        return compute_ngram_similarity(h, r, embedder, cfg, hyp_embeds, ref_embeds)

    if matching in ("soft", "greedy"):
        used_ref = set()
        sum_sim  = 0.0
        matched  = 0
        for h in hyp_ngrams:
            best_sim, best_j = 0.0, -1
            for j, r in enumerate(ref_ngrams):
                if j not in used_ref:
                    s = _sim(h, r)
                    if s > best_sim:
                        best_sim, best_j = s, j
            if best_j >= 0 and best_sim >= thr:
                used_ref.add(best_j)
                sum_sim += best_sim
                matched += 1
    else:
        sim_mat = np.array([[_sim(h, r) for r in ref_ngrams] for h in hyp_ngrams])
        row_ind, col_ind = linear_sum_assignment(-sim_mat)
        sum_sim = sum(sim_mat[i, j] for i, j in zip(row_ind, col_ind)
                      if sim_mat[i, j] >= thr)
        matched = sum(1 for i, j in zip(row_ind, col_ind)
                      if sim_mat[i, j] >= thr)

    precision = sum_sim / len(hyp_ngrams)
    recall    = sum_sim / len(ref_ngrams)   # v7: fixed, always ref count
    f1 = (2.0 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return float(f1), int(matched), int(matched)


# ---------------------------------------------------------------------------
# Per-head score
# ---------------------------------------------------------------------------

def compute_head_score(
    hyp_ngrams_by_order,
    ref_ngrams_by_order,
    embedder,
    matching:             str                   = "greedy",
    similarity_threshold: float                 = 0.63,
    max_n:                int                   = 4,
    order_weights:        Optional[List[float]] = None,
    hyp_embeds:           Optional[EmbedDict]   = None,
    ref_embeds:           Optional[EmbedDict]   = None,
    soft_patterns:        bool                  = False,
    cfg:                  SynGramConfig         = DEFAULT_CONFIG,
) -> float:
    """Score one matched head pair across all n-gram orders."""
    n_orders = min(max_n, len(hyp_ngrams_by_order), len(ref_ngrams_by_order))
    if n_orders <= 0:
        return 0.0

    _ow = (order_weights if order_weights is not None else cfg.order_weights)[:n_orders]
    active = [i for i in range(n_orders)
              if hyp_ngrams_by_order[i] and ref_ngrams_by_order[i]]
    if not active:
        return 0.0

    raw_w  = [_ow[i] for i in active]
    norm_w = [w / sum(raw_w) for w in raw_w]

    score = 0.0
    for idx, w in zip(active, norm_w):
        f1, _, _ = compute_precision_for_n(
            hyp_ngrams_by_order[idx], ref_ngrams_by_order[idx],
            embedder, matching, similarity_threshold,
            cfg.adaptive_threshold, idx + 1,
            hyp_embeds, ref_embeds, cfg,
        )
        score += w * f1
    return float(score)


# ---------------------------------------------------------------------------
# Pooled scoring  (backward compatibility)
# ---------------------------------------------------------------------------

def compute_syntactic_ngram(
    hyp_ngrams_all,
    ref_ngrams_all,
    embedder,
    matching:             str                   = "greedy",
    similarity_threshold: float                 = 0.63,
    adaptive_threshold:   bool                  = True,
    use_cache:            bool                  = True,
    max_n:                int                   = 4,
    order_weights:        Optional[List[float]] = None,
    return_debug:         bool                  = False,
    hyp_embeds:           Optional[EmbedDict]   = None,
    ref_embeds:           Optional[EmbedDict]   = None,
    cfg:                  SynGramConfig         = DEFAULT_CONFIG,
):
    """Pooled scoring kept for backward compatibility."""
    n_orders = min(max_n, len(hyp_ngrams_all), len(ref_ngrams_all))
    if n_orders <= 0:
        return (0.0, []) if return_debug else 0.0

    _ow = (order_weights if order_weights is not None else cfg.order_weights)[:n_orders]
    active = [i for i in range(n_orders)
              if hyp_ngrams_all[i] and ref_ngrams_all[i]]
    if not active:
        return (0.0, []) if return_debug else 0.0

    raw_w  = [_ow[i] for i in active]
    norm_w = [w / sum(raw_w) for w in raw_w]

    order_scores, debug_list = [], []
    for idx, w in zip(active, norm_w):
        f1, hm, rm = compute_precision_for_n(
            hyp_ngrams_all[idx], ref_ngrams_all[idx],
            embedder, matching, similarity_threshold,
            adaptive_threshold, idx + 1, hyp_embeds, ref_embeds, cfg,
        )
        order_scores.append(f1)
        if return_debug:
            thr = (_adaptive_threshold(similarity_threshold, idx + 1)
                   if adaptive_threshold else similarity_threshold)
            debug_list.append({
                "order": idx + 1, "score": f1,
                "hyp_matches": hm, "ref_matches": rm,
                "threshold": thr, "weight": w,
            })

    final = sum(w * s for w, s in zip(norm_w, order_scores))
    return (float(final), debug_list) if return_debug else float(final)