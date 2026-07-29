"""
syntactic_ngrams.py

Extracts syntactic n-grams (dependency upward paths) from parsed sentences.

UPDATED (COMPATIBLE VERSION):
- Keeps original upward-path extraction
- Adds wrapper for GRIS-SynGram STAR n-grams
- Safe to use with Evaluate_new.py
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Set, Any

from gris.parser import DepSentence, DepToken
from gris.model_cache import get_stanza_parser
from gris.ngram_scorer import extract_dep_star_ngrams_from_deptokens

# =========================
# LANGUAGE SETTINGS
# =========================

SKIP_CASE_LANGUAGES = {
    "en", "de", "nl", "da", "no", "sv",
    "fr", "es", "pt", "it", "ro",
    "zh", "ja", "ko", "vi", "th"
}

MORPHOLOGICAL_CASE_LANGUAGES = {
    "fi", "hu", "tr", "et",
    "ru", "pl", "cs", "uk", "bg",
    "ar", "he",
    "hi", "bn", "ta", "te"
}

# =========================
# CORE HELPERS
# =========================

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


def should_skip_case_marker(lang: str, tok: DepToken, id2tok: Dict[int, DepToken]) -> bool:
    rel = (tok.deprel or "dep").lower()

    if rel == "aux:pass":
        return True

    if rel == "case" and tok.head in id2tok:
        head = id2tok[tok.head]
        head_rel = (head.deprel or "dep").lower()
        lang_base = (lang or "en").split("_")[0].lower()

        if lang_base in MORPHOLOGICAL_CASE_LANGUAGES:
            return False

        if lang_base in SKIP_CASE_LANGUAGES:
            if head_rel in {"obl", "obl:agent", "obl:arg", "obl:tmod", "obl:npmod", "obl:unmarked"}:
                return True

        if head_rel == "obl:agent":
            return True

    return False


# =========================
# ORIGINAL UPWARD N-GRAMS
# =========================

def extract_syntactic_ngrams(
    dep_sentence: DepSentence,
    max_n: int = 3,
    lang: str = "en",
    verb_only_lemmas: bool = True,
    collapse_roles_flag: bool = True,
) -> Tuple[List[Tuple[str, List[str]]], int]:

    ngrams: List[Tuple[str, List[str]]] = []
    id2tok: Dict[int, DepToken] = {t.id: t for t in dep_sentence.tokens}

    skip_ids: Set[int] = set()
    num_case_markers = 0

    for tok in dep_sentence.tokens:
        if should_skip_case_marker(lang, tok, id2tok):
            skip_ids.add(tok.id)
            if (tok.deprel or "").lower() == "case":
                num_case_markers += 1

    def maybe_add_lemma(t: DepToken, s: Set[str]):
        lemma = (t.lemma or t.text or "").strip().lower()
        if not lemma:
            return
        if verb_only_lemmas:
            if (t.upos or "").upper() == "VERB":
                s.add(lemma)
        else:
            s.add(lemma)

    def walk_upward(curr: DepToken, patterns: List[str], lemmas: Set[str]):
        if curr.head == 0 or curr.head not in id2tok:
            return

        head = id2tok[curr.head]

        rel = (curr.deprel or "dep").lower()
        rel = collapse_core_roles(rel) if collapse_roles_flag else rel

        edge_pattern = f"{(curr.upos or 'X')}/{rel}->{(head.upos or 'X')}"
        new_patterns = patterns + [edge_pattern]

        new_lemmas = set(lemmas)
        maybe_add_lemma(head, new_lemmas)

        ngrams.append(("/".join(new_patterns), sorted(new_lemmas)))

        if len(new_patterns) < max_n:
            walk_upward(head, new_patterns, new_lemmas)

    for tok in dep_sentence.tokens:
        if tok.head != 0 and tok.id not in skip_ids:
            init: Set[str] = set()
            maybe_add_lemma(tok, init)
            walk_upward(tok, [], init)

    return ngrams, num_case_markers


# =========================
# ⭐ NEW: GRIS-SYNGRAM COMPATIBLE WRAPPER
# =========================

def sentence_to_dependency_syntactic_ngrams_and_tokens(
    sent: str,
    lang: str,
    max_n: int = 4,
    lemma_content_only: bool = True,
    voice_normalize: bool = True,
    collapse_roles: bool = True,
    verb_only_lemmas: bool = False,
):
    """
    REQUIRED by GRIS-SynGram pipeline.
    Produces STAR n-grams + tokens.
    """

    parser = get_stanza_parser(lang)
    dep_sents = parser.parse([sent])

    if not dep_sents:
        return ([[] for _ in range(max_n)], [])

    tokens = dep_sents[0].tokens

    ngrams = extract_dep_star_ngrams_from_deptokens(
        tokens,
        lang=lang,
        max_n=max_n,
        lemma_content_only=lemma_content_only,
        include_neg_feature=True,
        voice_normalize=voice_normalize,
        collapse_roles=collapse_roles,
        verb_only_lemmas=verb_only_lemmas,
    )

    return ngrams, tokens


# =========================
# BATCH EXTRACTION
# =========================

def extract_from_corpus(
    dep_sentences: List[DepSentence],
    max_n: int = 3,
    lang: str = "en",
    verb_only_lemmas: bool = True,
    collapse_roles: bool = True,
):
    return [
        extract_syntactic_ngrams(
            s,
            max_n=max_n,
            lang=lang,
            verb_only_lemmas=verb_only_lemmas,
            collapse_roles_flag=collapse_roles,
        )
        for s in dep_sentences
    ]