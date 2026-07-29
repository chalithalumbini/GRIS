"""
ngram_scorer.py - GRIS-SynGram scorer  (v7)

CHANGES v7
----------
- Extraction duplication removed: extract_dep_star_ngrams_per_head is now
  imported directly from ngram_extractor.py. No extraction logic here.
- SynGramConfig drives all parameters; passed through the full call chain.
- Bilateral depth-decayed importance weights (geometric mean for pair weight).
- Hungarian head matching on importance-weighted similarity matrix.
- Unmatched ref heads add to denominator (coverage penalty).
- Passive sentence-level penalty removed (voice normalisation in extractor
  handles this at pattern level).
- Negation sentence-level penalty retained.
- Debug mode (return_debug=True in score_sentence_per_head) outputs:
    matched_heads, per_head_scores, penalties_fired, dropped_deps
- sentence_has_tense_mismatch dead code removed from scoring path.
- Parser and embedding outputs cached per sentence string.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gris.model_cache import get_embedder, get_stanza_parser

# All extraction from single source of truth
from gris.ngram_extractor import (
    SynGramConfig,
    DEFAULT_CONFIG,
    HeadNGrams,
    extract_dep_star_ngrams_per_head,
    CONTENT_UPOS,
)

from gris.ngram_matcher import (
    compute_head_score,
    compute_syntactic_ngram,
    LanguageConfig,
    clear_sim_cache,
    _cos01,
    EmbedDict,
)

from gris.shared_utils import sentence_has_negation, depth_gated_unmatched_weight

NGram = Tuple[str, List[str], List[Any]]


# =============================================================================
# LANGUAGE ADAPTATION
# =============================================================================

_MORPHOLOGICAL_LANGS = {
    "ru", "uk", "cs", "pl", "bg",
    "fi", "hu", "et", "tr",
    "ar", "he",
}


def _adapt_config_for_lang(cfg: SynGramConfig, lang: str) -> SynGramConfig:
    """
    Return a cfg adapted for morphological languages:
      - max_n capped at 3 (richer morphology compensates for fewer arg slots)
      - soft_patterns flag set (caller checks this)
    Leaves all other fields untouched.
    """
    lb = (lang or "en").split("_")[0].lower()
    if lb in _MORPHOLOGICAL_LANGS:
        from dataclasses import replace
        return replace(cfg, max_n=min(cfg.max_n, 3))
    return cfg


# =============================================================================
# HEAD IMPORTANCE  (depth-decayed, bilateral)
# =============================================================================

def _build_depth_map(tokens, id2t: Dict[int, Any]) -> Dict[int, int]:
    """Compute depth of each token from root. Handles cycles (max 20 steps)."""
    depth_map: Dict[int, int] = {}
    for t in tokens:
        tid, depth, current, visited = t.id, 0, t.id, set()
        while current in id2t and depth < 20:
            if current in visited:
                break
            visited.add(current)
            parent = getattr(id2t[current], "head", None)
            if parent is None or parent == 0:
                break
            depth  += 1
            current = parent
        depth_map[tid] = depth
    return depth_map


def _head_importance(head_id: int, depth_map: Dict[int, int]) -> float:
    """
    Harmonic depth decay:  w = 1 / (1 + depth)

    depth 0 → 1.000  (root)
    depth 1 → 0.500
    depth 2 → 0.333
    depth 3 → 0.250
    depth 4 → 0.200
    ...
    Genuinely continuous; no arbitrary floor or tier breaks.
    """
    depth = depth_map.get(head_id, 3)
    return 1.0 / (1.0 + depth)


def _build_importance(
    head_ids: List[int],
    tokens,
    id2t: Dict[int, Any],
) -> Dict[int, float]:
    depth_map = _build_depth_map(tokens, id2t)
    return {h_id: _head_importance(h_id, depth_map) for h_id in head_ids}


# =============================================================================
# EMBEDDING DICT
# =============================================================================

def _build_embed_dict(tokens, embedder) -> EmbedDict:
    """Build token_id -> contextualised vector via embed_tokens()."""
    if hasattr(embedder, "embed_tokens"):
        try:
            return embedder.embed_tokens(tokens)
        except Exception:
            pass
    result: EmbedDict = {}
    texts = [
        (getattr(t, "text", None) or getattr(t, "lemma", "") or "")
        for t in tokens
    ]
    if texts:
        try:
            vecs = embedder.encode(texts)
            for t, v in zip(tokens, vecs):
                result[t.id] = np.asarray(v, dtype=np.float32)
        except Exception:
            pass
    return result


# =============================================================================
# HEAD MATCHING  (Hungarian, importance-weighted)
# =============================================================================

def _match_heads_hungarian(
    hyp_head_ids:   List[int],
    ref_head_ids:   List[int],
    importance_hyp: Dict[int, float],
    importance_ref: Dict[int, float],
    hyp_embeds:     EmbedDict,
    ref_embeds:     EmbedDict,
) -> Dict[int, Optional[int]]:
    """
    Globally optimal head matching via Hungarian algorithm.

    Pair weight  = geometric_mean(w_hyp, w_ref) = sqrt(w_hyp * w_ref)
    Cost matrix  = -(sim * pair_weight)
    Assignment   = argmin sum(cost)   one-to-one constraint

    Geometric mean is 0 if either weight is 0, equals arithmetic mean when
    both weights are equal, and penalises mismatched importance more strongly.
    """
    n_hyp, n_ref = len(hyp_head_ids), len(ref_head_ids)
    if n_hyp == 0 or n_ref == 0:
        return {h: None for h in hyp_head_ids}

    sim_mat    = np.zeros((n_hyp, n_ref))
    weight_mat = np.zeros((n_hyp, n_ref))

    for i, h_id in enumerate(hyp_head_ids):
        h_vec = hyp_embeds.get(h_id)
        for j, r_id in enumerate(ref_head_ids):
            r_vec = ref_embeds.get(r_id)
            if h_vec is not None and r_vec is not None:
                sim_mat[i, j] = _cos01(h_vec, r_vec)
            # Geometric mean for pair weight
            weight_mat[i, j] = np.sqrt(
                importance_hyp[h_id] * importance_ref[r_id]
            )

    row_ind, col_ind = linear_sum_assignment(-(sim_mat * weight_mat))

    matching: Dict[int, Optional[int]] = {h: None for h in hyp_head_ids}
    for i, j in zip(row_ind, col_ind):
        h_id  = hyp_head_ids[i]
        r_id  = ref_head_ids[j]
        h_vec = hyp_embeds.get(h_id)
        r_vec = ref_embeds.get(r_id)
        if h_vec is not None and r_vec is not None:
            matching[h_id] = r_id

    return matching


# =============================================================================
# PER-HEAD SENTENCE SCORING
# =============================================================================

def score_sentence_per_head(
    hyp_head_ngrams: HeadNGrams,
    ref_head_ngrams: HeadNGrams,
    hyp_tokens,
    ref_tokens,
    hyp_embeds:  EmbedDict,
    ref_embeds:  EmbedDict,
    embedder,
    cfg:         SynGramConfig = DEFAULT_CONFIG,
    return_debug: bool          = False,
) -> Tuple[float, Optional[Dict]]:
    """
    Per-head scoring with bilateral importance and depth-gated coverage penalty.

    Steps:
    1. Depth-decayed harmonic importance for BOTH sides independently.
    2. Hungarian matching: cost = -(sim * geometric_mean(w_hyp, w_ref)).
    3. Pair score weight  = geometric_mean(w_hyp, w_ref).
    4. Unmatched ref heads add depth-gated weight to denominator.
       Shallow heads (depth < 2): full importance added.
       Deep heads (depth >= 2): only 20% of importance added.
       Motivation: German→EN structural divergence means deep unmatched
       ref heads are usually clause-reshaping artefacts, not translation
       errors. Equal-weight coverage penalty was systematically hurting
       valid paraphrastic translations.
    5. base_score = sum(w_i * head_score_i) / (matched_w + gated_unmatched_ref_w)

    Debug output (when return_debug=True):
        matched_heads  : list of (hyp_id, ref_id, pair_weight, head_score)
        unmatched_ref  : list of (ref_id, ref_weight, gated_weight)
        penalties_fired: sentence-level penalties and factors applied
    """
    if not hyp_head_ngrams:
        return (0.0, {"matched_heads": [], "unmatched_ref": [],
                      "penalties_fired": []} if return_debug else None)

    id2t_hyp = {t.id: t for t in hyp_tokens}
    id2t_ref = {t.id: t for t in ref_tokens}

    hyp_head_ids = list(hyp_head_ngrams.keys())
    ref_head_ids = list(ref_head_ngrams.keys())

    # Step 1: bilateral importance
    importance_hyp = _build_importance(hyp_head_ids, hyp_tokens, id2t_hyp)
    importance_ref = _build_importance(ref_head_ids, ref_tokens, id2t_ref)

    # Step 2: optimal matching
    head_matching = _match_heads_hungarian(
        hyp_head_ids, ref_head_ids,
        importance_hyp, importance_ref,
        hyp_embeds, ref_embeds,
    )

    # Depth maps for coverage penalty gating
    depth_map_ref = _build_depth_map(ref_tokens, id2t_ref)

    # Step 3 & 4: score matched pairs, track unmatched ref
    matched_ref_ids: set  = set()
    weighted_sum          = 0.0
    weight_matched        = 0.0
    debug_heads           = []

    for h_id, r_id in head_matching.items():
        if r_id is None or r_id not in ref_head_ngrams:
            continue
        matched_ref_ids.add(r_id)

        pair_w = float(np.sqrt(importance_hyp[h_id] * importance_ref[r_id]))

        head_s = compute_head_score(
            hyp_head_ngrams[h_id],
            ref_head_ngrams[r_id],
            embedder,
            cfg.matching,
            cfg.similarity_threshold,
            cfg.max_n,
            cfg.order_weights,
            hyp_embeds,
            ref_embeds,
            cfg=cfg,
        )
        weighted_sum  += pair_w * head_s
        weight_matched += pair_w

        if return_debug:
            debug_heads.append({
                "hyp_id":     h_id,
                "ref_id":     r_id,
                "pair_weight": round(pair_w, 4),
                "head_score":  round(head_s, 4),
            })

    # Unmatched ref heads: depth-gated contribution to denominator
    unmatched_ref = [
        (r_id, importance_ref[r_id])
        for r_id in ref_head_ids
        if r_id not in matched_ref_ids
    ]
    unmatched_ref_weight = sum(
        depth_gated_unmatched_weight(r_id, w, depth_map_ref)
        for r_id, w in unmatched_ref
    )

    total_weight = weight_matched + unmatched_ref_weight
    base_score   = float(weighted_sum / total_weight) if total_weight > 0 else 0.0

    debug_info = None
    if return_debug:
        debug_info = {
            "matched_heads": debug_heads,
            "unmatched_ref": [
                {
                    "ref_id":      r_id,
                    "ref_weight":  round(w, 4),
                    "gated_weight": round(
                        depth_gated_unmatched_weight(r_id, w, depth_map_ref), 4),
                }
                for r_id, w in unmatched_ref
            ],
            "penalties_fired": [],   # sentence-level penalties added by caller
        }

    return base_score, debug_info


# =============================================================================
# SENTENCE PIPELINE  (parse + embed + extract)
# =============================================================================

def sentence_to_ngrams_and_embeds(
    sent:       str,
    lang:       str,
    embedder,
    cfg:        SynGramConfig = DEFAULT_CONFIG,
    parse_lang: str            = None,
) -> Tuple[HeadNGrams, List, EmbedDict]:
    """
    Parse sentence, extract per-head n-grams (from ngram_extractor),
    build contextualised embed dict.

    parse_lang: language for Stanza parser (text language).
                Defaults to lang if not supplied.
                For zh→en: lang="zh" (task direction, controls thresholds)
                           parse_lang="en" (text is English, use EN parser).
    lang:       task direction — controls n-gram extraction config and
                language-adaptive thresholds in cfg.
    """
    _parse_lang = parse_lang or lang
    parser    = get_stanza_parser(_parse_lang)
    dep_sents = parser.parse([sent])
    if not dep_sents:
        return {}, [], {}

    tokens     = dep_sents[0].tokens
    embed_dict = _build_embed_dict(tokens, embedder)

    # All extraction via ngram_extractor — no duplication here
    head_ngrams = extract_dep_star_ngrams_per_head(tokens, lang, cfg)

    return head_ngrams, tokens, embed_dict


# =============================================================================
# PUBLIC API
# =============================================================================

def compute_syntactic_ngram_metric(
    hyps:              List[str],
    refs:              List[str],
    lang:              str                   = "en",
    parse_lang:        str                   = None,
    embedding_type:    str                   = "transformer",
    model_name:        str                   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    cfg:               SynGramConfig         = DEFAULT_CONFIG,
    return_pair_scores: bool                  = False,
    neg_penalty_factor: float                 = None,
    penalty_mismatch_only: bool               = True,
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """
    Compute GRIS-SynGram corpus score (v7).

    lang:       task direction — controls thresholds, n-gram config, STRUCT_WEIGHT.
    parse_lang: language for Stanza parser (text language). Defaults to lang.
                For zh→en: lang="zh", parse_lang="en".
                Separates "what translation task is this" from "what language is the text".

    All other parameters driven by cfg (SynGramConfig).
    neg_penalty_factor overrides cfg.neg_sent_penalty if supplied.
    Language adaptation (max_n cap for morphological languages) applied
    automatically from lang.

    Per-sentence scoring:
    1. Parse + embed both sides (cached via sentence string).
    2. Extract per-head n-grams from ngram_extractor (single source).
    3. Bilateral depth-decayed importance (harmonic: 1/(1+depth)).
    4. Hungarian head matching (geometric mean pair weights).
    5. Per-head F1 with standard recall (len(ref)).
    6. Sentence score = weighted_sum / (matched_w + unmatched_ref_w).
    7. Negation mismatch multiplier only (passive handled at pattern level).
    """
    if not hyps or not refs:
        return (0.0, {"pairs": []}) if return_pair_scores else (0.0, None)
    if len(hyps) != len(refs):
        raise ValueError("hyps and refs must have same length")

    # Language-adapted config
    lang_cfg = _adapt_config_for_lang(cfg, lang)
    _lb      = (lang or "en").split("_")[0].lower()
    is_morph = _lb in _MORPHOLOGICAL_LANGS

    _neg_factor = neg_penalty_factor if neg_penalty_factor is not None \
                  else lang_cfg.neg_sent_penalty

    embedder  = get_embedder(embedding_type=embedding_type, model_name=model_name)
    pair_rows = []
    scores    = []

    for h, r in zip(hyps, refs):
        # Clear per-sentence similarity cache
        clear_sim_cache()

        hyp_head_ngrams, hyp_tokens, hyp_embeds = sentence_to_ngrams_and_embeds(
            h, lang, embedder, lang_cfg, parse_lang=parse_lang)
        ref_head_ngrams, ref_tokens, ref_embeds = sentence_to_ngrams_and_embeds(
            r, lang, embedder, lang_cfg, parse_lang=parse_lang)

        base_score, debug_info = score_sentence_per_head(
            hyp_head_ngrams, ref_head_ngrams,
            hyp_tokens, ref_tokens,
            hyp_embeds, ref_embeds,
            embedder, lang_cfg,
            return_debug=return_pair_scores,
        )

        # Sentence-level: negation mismatch only
        hyp_neg = sentence_has_negation(hyp_tokens) if hyp_tokens else False
        ref_neg = sentence_has_negation(ref_tokens) if ref_tokens else False
        neg_pen = _neg_factor if (
            (penalty_mismatch_only and hyp_neg != ref_neg) or
            (not penalty_mismatch_only and (hyp_neg or ref_neg))
        ) else 1.0

        final_score = float(base_score) * float(neg_pen)
        scores.append(final_score)

        if return_pair_scores:
            if debug_info:
                debug_info["penalties_fired"].append(
                    {"type": "neg_sent", "factor": float(neg_pen),
                     "hyp_neg": bool(hyp_neg), "ref_neg": bool(ref_neg)}
                )
            pair_rows.append({
                "score":       final_score,
                "base_score":  float(base_score),
                "neg_penalty": float(neg_pen),
                "hyp_neg":     bool(hyp_neg),
                "ref_neg":     bool(ref_neg),
                "n_hyp_heads": len(hyp_head_ngrams),
                "n_ref_heads": len(ref_head_ngrams),
                "debug":       debug_info,
            })

    mean_score = sum(scores) / len(scores) if scores else 0.0
    return (mean_score, {"pairs": pair_rows}) if return_pair_scores else (mean_score, None)


# =============================================================================
# CLI
# =============================================================================

def _read_lines(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def _strip_empty_pairs(hyps, refs):
    out_h, out_r = [], []
    for h, r in zip(hyps, refs):
        if not ((h or "").strip() == "" and (r or "").strip() == ""):
            out_h.append((h or "").strip())
            out_r.append((r or "").strip())
    return out_h, out_r


def main():
    ap = argparse.ArgumentParser(
        description="GRIS-SynGram v7 — bilateral importance + Hungarian matching")
    ap.add_argument("--hyp",  required=True)
    ap.add_argument("--ref",  required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--csv",  required=True)
    ap.add_argument("--max_n",     type=int,   default=4)
    ap.add_argument("--max_deps",  type=int,   default=3)
    ap.add_argument("--matching",              default="soft",
                    choices=["soft", "greedy", "hungarian"])
    ap.add_argument("--threshold",   type=float, default=None)
    ap.add_argument("--neg_penalty", type=float, default=None)
    ap.add_argument("--arg_pen",     type=float, default=0.20,
                    help="ARG count mismatch penalty")
    ap.add_argument("--neg_edge_pen", type=float, default=0.25,
                    help="neg:PART mismatch penalty")
    ap.add_argument("--order_weights", nargs="+", type=float,
                    default=[0.50, 0.25, 0.15, 0.10])
    ap.add_argument("--no_collapse_roles",     action="store_true")
    ap.add_argument("--no_voice_normalize",    action="store_true")
    ap.add_argument("--no_lemma_content_only", action="store_true")
    ap.add_argument("--verb_only_lemmas",      action="store_true")
    ap.add_argument("--debug_csv",             default=None,
                    help="Optional path for per-head debug CSV output")
    args = ap.parse_args()

    thr = args.threshold or LanguageConfig.get_recommended_threshold(args.lang)

    cfg = SynGramConfig(
        max_n                     = args.max_n,
        max_deps                  = args.max_deps,
        matching                  = args.matching,
        similarity_threshold      = thr,
        order_weights             = args.order_weights,
        arg_count_mismatch_penalty = args.arg_pen,
        neg_edge_penalty          = args.neg_edge_pen,
        collapse_roles            = not args.no_collapse_roles,
        voice_normalize           = not args.no_voice_normalize,
        lemma_content_only        = not args.no_lemma_content_only,
        verb_only_lemmas          = args.verb_only_lemmas,
    )

    hyps = _read_lines(args.hyp)
    refs = _read_lines(args.ref)
    if len(hyps) != len(refs):
        raise SystemExit(f"ERROR: hyp ({len(hyps)}) != ref ({len(refs)})")
    hyps, refs = _strip_empty_pairs(hyps, refs)

    corpus_score, debug = compute_syntactic_ngram_metric(
        hyps, refs,
        lang=args.lang,
        cfg=cfg,
        return_pair_scores=True,
        neg_penalty_factor=args.neg_penalty,
        penalty_mismatch_only=True,
    )

    print(f"GRIS-SynGram v7 corpus score ({args.lang}): {corpus_score:.6f}")
    pairs = (debug or {}).get("pairs", [])

    # Main CSV
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "hyp", "ref", "score", "base_score",
            "neg_penalty", "hyp_neg", "ref_neg",
            "n_hyp_heads", "n_ref_heads",
        ])
        for i, (h, r) in enumerate(zip(hyps, refs)):
            row = pairs[i] if i < len(pairs) else {}
            w.writerow([
                i, h, r,
                f"{row.get('score',      0.0):.6f}",
                f"{row.get('base_score', 0.0):.6f}",
                f"{row.get('neg_penalty',1.0):.6f}",
                row.get('hyp_neg',  False),
                row.get('ref_neg',  False),
                row.get('n_hyp_heads', 0),
                row.get('n_ref_heads', 0),
            ])
        w.writerow(["CORPUS_MEAN", "", "", f"{corpus_score:.6f}",
                    "", "", "", "", "", ""])

    print(f"Saved -> {args.csv}")

    # Optional per-head debug CSV
    if args.debug_csv:
        with open(args.debug_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "sent_id", "hyp_head_id", "ref_head_id",
                "pair_weight", "head_score",
                "neg_penalty_fired", "unmatched_ref_ids",
            ])
            for i, row in enumerate(pairs):
                dbg = row.get("debug") or {}
                matched   = dbg.get("matched_heads", [])
                unmatched = dbg.get("unmatched_ref", [])
                neg_fired = any(
                    p.get("type") == "neg_sent" and p.get("factor", 1.0) < 1.0
                    for p in dbg.get("penalties_fired", [])
                )
                unmatched_ids = ";".join(str(u["ref_id"]) for u in unmatched)
                for mh in matched:
                    w.writerow([
                        i,
                        mh["hyp_id"], mh["ref_id"],
                        mh["pair_weight"], mh["head_score"],
                        neg_fired, unmatched_ids,
                    ])
                if not matched:
                    w.writerow([i, "", "", "", "", neg_fired, unmatched_ids])
        print(f"Debug CSV -> {args.debug_csv}")


if __name__ == "__main__":
    main()