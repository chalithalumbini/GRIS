"""
ngram_extractor.py - Subtree syntactic n-gram extraction  (v8)

ARCHITECTURE CHANGE v8: STAR → SUBTREE PATH n-grams
----------------------------------------------------
The v7 STAR topology built patterns from a head and its DIRECT dependents
only (one level). This was too shallow to discriminate between valid
paraphrases and translation errors for German and Chinese, where the
predicate-argument structure spans multiple levels (verb → subject →
subject's modifier, verb → object → object's preposition etc).

v8 switches to SUBTREE PATH n-grams:

  For each content head H in the sentence:
    1. Collect the full subtree rooted at H (all descendants, depth-limited).
    2. Extract upward dependency PATHS from each leaf/node in the subtree
       back to H as the anchor.
    3. Each path of length k becomes a k-gram pattern:
         (leaf_UPOS/dep_rel → ... → H_UPOS)
       with the lemma of each node on the path stored alongside.

This is closely aligned with Liu & Gildea (2005) syntactic n-grams and
captures:
  - Full predicate-argument chains  (VERB→nsubj→NOUN→amod→ADJ)
  - Deep modification structures    (VERB→obj→NOUN→nmod→PROPN)
  - Negation scope within subtree
  - Passive constructions via voice normalisation

Benefits over STAR:
  - German V2 and head-final structures: subject and object appear at
    different surface positions but share the same upward path to the
    verb → correctly matches across topicalised and canonical orders.
  - Russian free word order: dep labels are stable; path patterns are
    identical for scrambled variants of the same proposition.
  - ZH→EN: deeper path patterns capture argument structure beyond the
    immediate dependents visible in STAR patterns.

Per-head grouping is preserved: each head H anchors a set of path n-grams
extracted from its subtree. The scorer still matches heads between
hypothesis and reference and scores them independently.

Public API (unchanged from v7 — all callers are compatible):
------------------------------------------------------------
extract_dep_star_ngrams_from_deptokens(tokens, lang, cfg)
extract_dep_star_ngrams_per_head(tokens, lang, cfg)
extract_head_grouped_ngrams(tokens, lang, cfg)
SynGramConfig
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
NGram     = Tuple[str, List[str], List[Any]]
HeadGroup = Tuple[Any, List[List[NGram]], List[Any]]
HeadNGrams = Dict[int, List[List[NGram]]]


# ---------------------------------------------------------------------------
# SynGramConfig
# ---------------------------------------------------------------------------

@dataclass
class SynGramConfig:
    """
    All tunable constants for GRIS-SynGram in one place.

    Extraction params
    -----------------
    max_n               : maximum path length (n-gram order 1..max_n)
    max_deps            : max content dependents per head considered per level
    max_subtree_depth   : maximum depth to recurse into subtree (default 3)
                          depth=1 reproduces STAR behaviour for ablation.
    lemma_content_only  : only include lemmas for content-UPOS tokens
    include_neg_feature : encode neg:PART flag in patterns
    voice_normalize     : normalise passive/active argument labels
    collapse_roles      : collapse core argument labels to ARG
    verb_only_lemmas    : only store lemmas for VERB heads

    Similarity / threshold params
    ------------------------------
    similarity_threshold    : base cosine threshold for n-gram matching
    adaptive_threshold      : apply per-order threshold reduction

    Importance / weighting params
    ------------------------------
    order_weights           : per-order weights [n=1, n=2, n=3, n=4]
                              renormalised over active orders at runtime

    Penalty params
    --------------
    arg_count_mismatch_penalty : fired when ARG slot count differs
    neg_edge_penalty           : fired when neg:PART in one pattern only
    neg_sent_penalty           : sentence-level negation mismatch multiplier

    Bonus params
    ------------
    exact_match_bonus          : applied to VERB/NOUN/PROPN exact lemma matches
    exact_match_upos           : UPOS set eligible for exact-match bonus
    """
    # Extraction
    max_n:               int   = 4
    max_deps:            int   = 3
    max_subtree_depth:   int   = 3    # NEW: how deep to recurse into subtree
    lemma_content_only:  bool  = True
    include_neg_feature: bool  = True
    voice_normalize:     bool  = True
    collapse_roles:      bool  = True
    verb_only_lemmas:    bool  = False

    # Similarity
    similarity_threshold: float = 0.55
    adaptive_threshold:   bool  = True

    # Order weights — bigram-dominant [0.25, 0.40, 0.25, 0.10]
    # n=2 path bigrams encode one dependency edge (head→dep or dep→head),
    # which is the minimal structural unit that distinguishes translations.
    # n=3 and n=4 capture two- and three-edge paths through the subtree.
    order_weights: List[float] = field(
        default_factory=lambda: [0.25, 0.40, 0.25, 0.10]
    )

    # Penalties
    arg_count_mismatch_penalty: float = 0.20
    neg_edge_penalty:           float = 0.25
    neg_sent_penalty:           float = 0.70

    # Bonus
    exact_match_bonus: float = 0.05
    exact_match_upos:  frozenset = field(
        default_factory=lambda: frozenset({"VERB", "NOUN", "PROPN"})
    )

    # Matching
    matching: str = "soft"

    # Language hint
    lang: str = "en"


DEFAULT_CONFIG = SynGramConfig()


# ---------------------------------------------------------------------------
# Language sets
# ---------------------------------------------------------------------------

CONTENT_UPOS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}

# Modal/auxiliary verbs to include in patterns when they are children of VERB heads.
# These are semantically core (negation, modality, tense) and their absence
# represents real translation errors. Previously excluded via CONTENT_UPOS filter,
# causing modal errors (missing "kann", "muss", "soll") to be invisible to GRIS.
# Evidence: bathtub example MQM=-10, GRIS=0.875 because "kann" was excluded.
# Fix: include AUX in subtree path extraction when head is VERB.
MODAL_DEPRELS = {"aux", "aux:pass"}   # dep labels that mark modals/auxiliaries
MODAL_INCLUDE_UPOS = {"AUX"}          # UPOS tags to include for modal detection

SKIP_CASE_LANGUAGES = {
    "en", "de", "nl", "da", "no", "sv",
    "fr", "es", "pt", "it", "ro",
    "zh", "ja", "ko", "vi", "th",
}

MORPHOLOGICAL_CASE_LANGUAGES = {
    "fi", "hu", "tr", "et",
    "ru", "pl", "cs", "uk", "bg",
    "ar", "he",
    "hi", "bn", "ta", "te",
}

_DEPREL_MAP = {
    "agent": "obl:agent",
}

DEFAULT_MAX_DEPTH = 3

# ---------------------------------------------------------------------------
# Dependent role priority
# ---------------------------------------------------------------------------
_REL_PRIORITY: Dict[str, int] = {
    "nsubj":        0,
    "nsubj:pass":   0,
    "csubj":        0,
    "csubj:pass":   0,
    "obj":          1,
    "iobj":         1,
    "ccomp":        2,
    "xcomp":        2,
    "obl":          3,
    "obl:agent":    3,
    "obl:arg":      3,
    "obl:tmod":     4,
    "obl:npmod":    4,
    "obl:unmarked": 4,
    "advmod":       5,
    "amod":         6,
    "nmod":         7,
    "nummod":       8,
    "conj":         9,
    "appos":       10,
}
_REL_PRIORITY_DEFAULT = 20


def _dep_priority(tok) -> int:
    rel = (getattr(tok, "deprel", None) or "dep").lower()
    return _REL_PRIORITY.get(rel, _REL_PRIORITY_DEFAULT)


# ---------------------------------------------------------------------------
# Token-level helpers
# ---------------------------------------------------------------------------

def _lemma_tok(t) -> str:
    return (getattr(t, "lemma", None) or getattr(t, "text", "") or "").strip().lower()


def _upos_tok(t) -> str:
    return (getattr(t, "upos", None) or "X").upper()


def _deprel_tok_raw(t) -> str:
    rel = (getattr(t, "deprel", None) or "dep").lower()
    return _DEPREL_MAP.get(rel, rel)


def _lang_base(lang: str) -> str:
    return (lang or "en").split("_")[0].lower()


def collapse_core_roles(label: str) -> str:
    label = (label or "dep").lower()
    CORE = {
        "nsubj", "nsubj:pass",
        "obj", "iobj",
        "obl", "obl:arg",
        "obl:agent",
        "obl:tmod", "obl:npmod",
        "obl:unmarked",
        "csubj", "csubj:pass",
    }
    return "ARG" if label in CORE else label


def _is_passive_head(head_tok) -> bool:
    feats = getattr(head_tok, "feats", None) or {}
    return bool(isinstance(feats, dict) and feats.get("Voice") == "Pass")


def _should_skip_case_marker(lang: str, tok, id2t: Dict[int, Any]) -> bool:
    rel = (getattr(tok, "deprel", None) or "dep").lower()
    if rel == "aux:pass":
        return True
    if rel == "case" and getattr(tok, "head", 0) in id2t:
        head     = id2t[tok.head]
        head_rel = (getattr(head, "deprel", None) or "dep").lower()
        lb       = _lang_base(lang)
        if lb in MORPHOLOGICAL_CASE_LANGUAGES:
            return False
        if lb in SKIP_CASE_LANGUAGES:
            if head_rel in {"obl", "obl:agent", "obl:arg",
                            "obl:tmod", "obl:npmod", "obl:unmarked"}:
                return True
        if head_rel == "obl:agent":
            return True
    return False


def _normalize_rel_for_voice(head_is_passive: bool, rel: str) -> str:
    rel = (rel or "dep").lower()
    if head_is_passive:
        if rel in ("nsubj:pass", "csubj:pass"): return "obj"
        elif rel in ("obl:agent", "agent"):      return "nsubj"
        elif rel == "aux:pass":                  return "aux"
    else:
        if rel in ("nsubj:pass", "csubj:pass"): return "nsubj"
        elif rel in ("obl:agent", "agent"):      return "obl"
    return rel


# ---------------------------------------------------------------------------
# Shared internal helpers
# ---------------------------------------------------------------------------

def _build_children(
    tokens,
    lang: str,
) -> Tuple[Dict[int, Any], Dict[int, List[int]]]:
    """Build id→token map and children dict.

    Includes AUX dependents of VERB heads (modals, tense auxiliaries) so
    that modal errors like missing 'kann'/'muss' are visible to GRIS.
    Previously AUX was excluded via CONTENT_UPOS, causing MQM=-10 segments
    where the only error is a missing modal to score ~0.875 (too high).

    Case markers are still hard-skipped via _should_skip_case_marker.
    """
    id2t = {t.id: t for t in tokens}
    children: Dict[int, List[int]] = defaultdict(list)
    for t in tokens:
        if t.head and t.head != 0:
            if _should_skip_case_marker(lang, t, id2t):
                continue
            rel  = (getattr(t, "deprel", None) or "dep").lower()
            upos = _upos_tok(t)
            # Always include AUX children of VERB heads (modals, tense markers).
            # These encode modality, negation scope, and tense — semantically core.
            # Excluding them made modal errors (missing "kann", "muss") invisible.
            if upos in MODAL_INCLUDE_UPOS and rel in MODAL_DEPRELS:
                head = id2t.get(t.head)
                if head and _upos_tok(head) == "VERB":
                    children[t.head].append(t.id)
                    continue
            children[t.head].append(t.id)
    return id2t, children


def _select_content_deps(
    dep_ids: List[int],
    id2t: Dict[int, Any],
    max_deps: int,
    head_upos: str = "X",
) -> List[int]:
    """Select up to max_deps content dependents ranked by role priority.

    For VERB heads, AUX dependents (modals) are included alongside
    content-UPOS tokens. They are ranked at priority 0.5 (between
    nsubj and obj) since modality is semantically critical.
    """
    content = []
    for did in dep_ids:
        tok_upos = _upos_tok(id2t[did])
        if tok_upos in CONTENT_UPOS:
            content.append(did)
        elif (tok_upos in MODAL_INCLUDE_UPOS
              and head_upos == "VERB"
              and (getattr(id2t[did], "deprel", "") or "").lower() in MODAL_DEPRELS):
            content.append(did)   # include modal AUX for VERB heads
    if len(content) <= max_deps:
        return sorted(content)
    ranked = sorted(content, key=lambda did: (_dep_priority(id2t[did]), did))
    return sorted(ranked[:max_deps])


# ---------------------------------------------------------------------------
# v8: Subtree path n-gram extraction
# ---------------------------------------------------------------------------

def _collect_subtree_paths(
    anchor_id:  int,
    id2t:       Dict[int, Any],
    children:   Dict[int, List[int]],
    cfg:        SynGramConfig,
) -> List[List[Any]]:
    """
    Collect all upward paths from nodes in the subtree rooted at anchor_id
    back to anchor_id.

    Each path is a list of token objects [leaf, ..., anchor].
    Path length 1 = anchor only (unigram).
    Path length 2 = direct child → anchor (bigram, equivalent to STAR edge).
    Path length 3 = grandchild → child → anchor (trigram).
    Path length 4 = great-grandchild → ... → anchor (4-gram).

    Depth is limited to cfg.max_subtree_depth to prevent combinatorial
    explosion on deep trees (long German subordinate clauses).

    At each level, content dependents are ranked by role priority and
    capped at cfg.max_deps, so the most linguistically important
    branches are always explored.

    Returns list of paths (each path is a list of token objects,
    ordered from leaf to anchor).
    """
    paths: List[List[Any]] = []

    # Path of length 1: anchor itself
    anchor = id2t.get(anchor_id)
    if anchor is None:
        return paths
    paths.append([anchor])

    def recurse(current_id: int, path_so_far: List[Any], depth: int) -> None:
        if depth >= cfg.max_subtree_depth:
            return
        dep_ids  = children.get(current_id, [])
        cur_upos = _upos_tok(id2t[current_id]) if current_id in id2t else "X"
        # Pass head_upos so AUX modals are included for VERB heads
        selected = _select_content_deps(dep_ids, id2t, cfg.max_deps, head_upos=cur_upos)
        for did in selected:
            tok = id2t.get(did)
            if tok is None:
                continue
            new_path = [tok] + path_so_far   # leaf → ... → anchor
            if len(new_path) <= cfg.max_n:
                paths.append(new_path)
            recurse(did, new_path, depth + 1)

    recurse(anchor_id, [anchor], 0)
    return paths


def _path_to_ngram(
    path:            List[Any],
    id2t:            Dict[int, Any],
    children:        Dict[int, List[int]],
    cfg:             SynGramConfig,
    anchor_is_passive: bool,
    anchor_has_neg:  bool,
) -> NGram:
    """
    Convert a token path [leaf, ..., anchor] into an NGram tuple.

    Pattern format:
      anchor_UPOS[neg:PART?,dep_k:UPOS_k/.../dep_1:UPOS_1]

    Where dep_i:UPOS_i is the edge from token i+1 → token i on the path
    (i.e. the dep label on the non-anchor end of each edge, read leaf→anchor).

    This encodes:
      - The anchor's UPOS (head of subtree)
      - The sequence of dependency labels along the path
      - Whether negation is present at the anchor
      - Voice normalisation for passive heads

    Lemmas: all content-UPOS tokens on the path, anchor first.
    Tokens: all tokens on the path.
    """
    anchor = path[-1]
    anchor_upos = _upos_tok(anchor)

    parts: List[str] = []
    if anchor_has_neg:
        parts.append("neg:PART")

    # Build edge labels from leaf to anchor (path[0] → path[1] → ... → path[-1])
    # Each edge label is the deprel of the non-anchor node in that edge
    for i in range(len(path) - 1):
        tok = path[i]  # the child in this edge
        rel_raw = (getattr(tok, "deprel", "") or "dep").lower()
        if cfg.voice_normalize:
            rel = _normalize_rel_for_voice(anchor_is_passive, rel_raw)
        else:
            rel = _DEPREL_MAP.get(rel_raw, rel_raw)
        if cfg.collapse_roles:
            rel = collapse_core_roles(rel)
        dep_upos = _upos_tok(tok)
        parts.append(f"{rel}:{dep_upos}")

    pattern = f"{anchor_upos}[{','.join(sorted(parts))}]"

    # Lemmas: anchor first, then content deps along path
    lemmas: List[str] = [_lemma_tok(anchor)]
    for tok in path[:-1]:  # all except anchor
        if cfg.verb_only_lemmas:
            if _upos_tok(tok) == "VERB":
                lemmas.append(_lemma_tok(tok))
        elif cfg.lemma_content_only:
            if _upos_tok(tok) in CONTENT_UPOS:
                lemmas.append(_lemma_tok(tok))
        else:
            lemmas.append(_lemma_tok(tok))

    if len(lemmas) == 1 and len(path) > 1:
        lemmas.append(_lemma_tok(path[0]))

    return (pattern, lemmas, path)


def _build_ngrams_for_head_subtree(
    head_id:  int,
    id2t:     Dict[int, Any],
    children: Dict[int, List[int]],
    cfg:      SynGramConfig,
) -> List[List[NGram]]:
    """
    Build subtree path n-grams for a single anchor head.

    Returns List[List[NGram]] indexed by order 0..cfg.max_n-1.

    Compared to v7 _build_ngrams_for_head:
      - Paths go through the FULL subtree (up to max_subtree_depth levels)
        rather than only immediate children.
      - A path of length k becomes an order-k n-gram.
      - Passive and negation are detected from direct children of the anchor
        (same as v7) but the patterns themselves can span multiple levels.
    """
    head = id2t.get(head_id)
    if head is None:
        return [[] for _ in range(cfg.max_n)]

    direct_dep_ids = children.get(head_id, [])

    # Passive detection — from direct children (same as v7)
    anchor_is_passive = False
    if cfg.voice_normalize:
        anchor_is_passive = _is_passive_head(head)
        if not anchor_is_passive:
            for did in direct_dep_ids:
                rel0 = (getattr(id2t[did], "deprel", "") or "").lower()
                if rel0 in ("aux:pass", "nsubj:pass", "csubj:pass"):
                    anchor_is_passive = True
                    break

    # Negation detection — from direct children
    anchor_has_neg = False
    if cfg.include_neg_feature:
        for did in direct_dep_ids:
            if (getattr(id2t[did], "deprel", "") or "").lower() == "neg":
                anchor_has_neg = True
                break

    # Collect all paths from nodes in subtree back to anchor
    paths = _collect_subtree_paths(head_id, id2t, children, cfg)

    # Bucket by path length (= n-gram order)
    out: List[List[NGram]] = [[] for _ in range(cfg.max_n)]
    seen_patterns: set = set()  # deduplicate identical patterns

    for path in paths:
        order = len(path)  # 1=unigram, 2=bigram, etc.
        if order > cfg.max_n:
            continue
        ngram = _path_to_ngram(
            path, id2t, children, cfg, anchor_is_passive, anchor_has_neg
        )
        # Deduplicate: same pattern string at same order (can occur in
        # symmetric subtrees with identical dep labels)
        key = (order, ngram[0])
        if key not in seen_patterns:
            seen_patterns.add(key)
            out[order - 1].append(ngram)

    return out


# ---------------------------------------------------------------------------
# Public extractors  (API identical to v7)
# ---------------------------------------------------------------------------

def extract_dep_star_ngrams_per_head(
    tokens,
    lang:  str,
    cfg:   SynGramConfig = DEFAULT_CONFIG,
    max_n:               Optional[int]  = None,
    lemma_content_only:  Optional[bool] = None,
    include_neg_feature: Optional[bool] = None,
    voice_normalize:     Optional[bool] = None,
    collapse_roles:      Optional[bool] = None,
    verb_only_lemmas:    Optional[bool] = None,
    max_deps:            Optional[int]  = None,
) -> HeadNGrams:
    """
    Extract subtree path n-grams grouped by anchor head_id.

    v8: Uses full subtree paths (depth-limited to cfg.max_subtree_depth)
    instead of v7's one-level STAR patterns.

    Returns:
        dict: head_id -> List[List[NGram]]
              outer list index = n-gram order (0=unigrams, 1=bigrams, …)
    """
    if not tokens:
        return {}

    if any(v is not None for v in [
            max_n, lemma_content_only, include_neg_feature,
            voice_normalize, collapse_roles, verb_only_lemmas, max_deps]):
        from dataclasses import replace
        cfg = replace(
            cfg,
            **{k: v for k, v in {
                "max_n":               max_n,
                "lemma_content_only":  lemma_content_only,
                "include_neg_feature": include_neg_feature,
                "voice_normalize":     voice_normalize,
                "collapse_roles":      collapse_roles,
                "verb_only_lemmas":    verb_only_lemmas,
                "max_deps":            max_deps,
            }.items() if v is not None}
        )

    id2t, children = _build_children(tokens, lang)

    def is_content_id(tid: int) -> bool:
        return _upos_tok(id2t[tid]) in CONTENT_UPOS

    result: HeadNGrams = {}
    for head_id in list(id2t.keys()):
        if not is_content_id(head_id):
            continue
        # Only extract for heads that have at least one dependent
        # (pure leaf content words produce only unigrams — still useful
        # as they carry the head's own semantic representation)
        ngrams = _build_ngrams_for_head_subtree(head_id, id2t, children, cfg)
        if any(ng_list for ng_list in ngrams):
            result[head_id] = ngrams

    return result


def extract_dep_star_ngrams_from_deptokens(
    tokens,
    lang:                str,
    cfg:                 SynGramConfig = DEFAULT_CONFIG,
    max_n:               Optional[int]  = None,
    lemma_content_only:  Optional[bool] = None,
    include_neg_feature: Optional[bool] = None,
    voice_normalize:     Optional[bool] = None,
    collapse_roles:      Optional[bool] = None,
    verb_only_lemmas:    Optional[bool] = None,
) -> List[List[NGram]]:
    """
    Extract subtree path n-grams pooled across all heads.
    Used by pooled scoring mode and standalone CLI.
    """
    if not tokens:
        _n = max_n if max_n is not None else cfg.max_n
        return [[] for _ in range(_n)]

    per_head = extract_dep_star_ngrams_per_head(
        tokens, lang, cfg,
        max_n=max_n,
        lemma_content_only=lemma_content_only,
        include_neg_feature=include_neg_feature,
        voice_normalize=voice_normalize,
        collapse_roles=collapse_roles,
        verb_only_lemmas=verb_only_lemmas,
    )

    _n = max_n if max_n is not None else cfg.max_n
    out: List[List[NGram]] = [[] for _ in range(_n)]
    for ng_by_order in per_head.values():
        for i, ng_list in enumerate(ng_by_order):
            if i < _n:
                out[i].extend(ng_list)
    return out


def extract_head_grouped_ngrams(
    tokens,
    lang: str,
    cfg:  SynGramConfig = DEFAULT_CONFIG,
) -> List[HeadGroup]:
    """
    Legacy tree-mode extractor for backward compatibility.
    Returns List[(head_token, List[List[NGram]], [head_token])].
    """
    per_head = extract_dep_star_ngrams_per_head(tokens, lang, cfg)
    id2t = {t.id: t for t in tokens}
    result: List[HeadGroup] = []
    for head_id, ngrams in per_head.items():
        head_tok = id2t.get(head_id)
        if head_tok is not None:
            result.append((head_tok, ngrams, [head_tok]))
    return result