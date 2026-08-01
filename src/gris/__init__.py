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
    from gris import compute_DepScore_emb

    score = compute_DepScore_emb(hyps=["..."], refs=["..."], lang="de")

Interpretability (without the dashboard)
-----------------------------------------
Pass ``explain=True`` to print a step-by-step breakdown of exactly how each
score was derived — matched dependency edges, precision/recall/Fβ, the
language blend weight, and which sentence-level penalties fired:

    score = compute_DepScore_emb(hyps=["..."], refs=["..."], lang="de",
                                  explain=True)

Or pass ``return_details=True`` to get the breakdown back as data, to
inspect programmatically or pretty-print yourself with ``explain_dep_score``:

    score, details = compute_DepScore_emb(hyps=["..."], refs=["..."],
                                           lang="de", return_details=True)
    explain_dep_score(details[0])

Heavy resources (Stanza pipelines, embedders) are cached per-process via
``gris.model_cache`` — see that module if you need to pre-warm models.
"""

from .scorer import compute_DepScore_emb
from .matcher import explain_dep_score
from .ngram_scorer import compute_syntactic_ngram_metric
from .ngram_extractor import SynGramConfig, DEFAULT_CONFIG
from .parser import StanzaDependencyParser, DepSentence, DepToken
from .embedder import EmbeddingModel

__version__ = "2.4.0"

__all__ = [
    "compute_DepScore_emb",
    "explain_dep_score",
    "compute_syntactic_ngram_metric",
    "SynGramConfig",
    "DEFAULT_CONFIG",
    "StanzaDependencyParser",
    "DepSentence",
    "DepToken",
    "EmbeddingModel",
]
