"""
GRIS — Interpretable Machine Translation Evaluation
=====================================================

Two complementary, linguistically-grounded MT evaluation metrics:

- **GRIS-DepScore**  — dependency-structure matching via the Hungarian
  algorithm (see ``gris.scorer.compute_DepScore_emb``).
- **GRIS-SynGram**   — subtree path n-gram matching (see
  ``gris.ngram_scorer.compute_syntactic_ngram_metric``).

Quick start
-----------
    from gris import compute_DepScore_emb, compute_syntactic_ngram_metric

    dep_score = compute_DepScore_emb(hyps=["..."], refs=["..."], lang="de")

Heavy resources (Stanza pipelines, embedders) are cached per-process via
``gris.model_cache`` — see that module if you need to pre-warm models.
"""

from .scorer import compute_DepScore_emb
from .ngram_scorer import compute_syntactic_ngram_metric
from .ngram_extractor import SynGramConfig, DEFAULT_CONFIG
from .parser import StanzaDependencyParser, DepSentence, DepToken
from .embedder import EmbeddingModel

__version__ = "2.4.0"

__all__ = [
    "compute_DepScore_emb",
    "compute_syntactic_ngram_metric",
    "SynGramConfig",
    "DEFAULT_CONFIG",
    "StanzaDependencyParser",
    "DepSentence",
    "DepToken",
    "EmbeddingModel",
]
