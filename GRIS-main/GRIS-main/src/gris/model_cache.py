"""model_cache.py

Centralized, process-local caching for heavyweight NLP resources.

This avoids repeated *initialization* (and the associated overhead) of:
  - Stanza dependency parser pipelines
  - SentenceTransformer / word-vector embedders
  - spaCy+benepar constituency parser
  - COMET checkpoint model

Most libraries already cache downloads on disk; the main win here is avoiding
repeated construction of pipelines/models within a single Python process.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from gris.parser import StanzaDependencyParser
from gris.embedder import EmbeddingModel
from gris.parser_constituency import ConstituencyParser


# ===============================
# Optional COMET
# ===============================
try:
    from comet import download_model, load_from_checkpoint  # type: ignore
    HAS_COMET = True
except Exception:
    HAS_COMET = False


# ===============================
# Stanza dependency parser cache
# ===============================
@lru_cache(maxsize=8)
def get_stanza_parser(lang: str = "en", use_gpu: bool = False) -> StanzaDependencyParser:
    """Get a cached StanzaDependencyParser per (lang, use_gpu)."""
    return StanzaDependencyParser(lang=lang, use_gpu=use_gpu)


# ===============================
# Embedding model cache
# ===============================
@lru_cache(maxsize=8)
def get_embedder(
    model_name: str = "glove-wiki-gigaword-300",
    embedding_type: str = "word",
    device: Optional[str] = None,
) -> EmbeddingModel:
    """Get a cached EmbeddingModel.

    Note: device is accepted for forward-compatibility. The current EmbeddingModel
    picks CUDA if available when device is None.
    """
    return EmbeddingModel(model_name=model_name, embedding_type=embedding_type, device=device)


# ===============================
# Constituency parser cache
# ===============================
@lru_cache(maxsize=4)
def get_constituency_parser(lang: str = "en") -> ConstituencyParser:
    """Get a cached ConstituencyParser per language."""
    return ConstituencyParser(lang=lang)


# ===============================
# COMET cache
# ===============================
@lru_cache(maxsize=1)
def get_comet_model():
    """Load and cache the COMET model checkpoint.

    Returns None if COMET isn't installed.
    """
    if not HAS_COMET:
        return None
    model_path = download_model("Unbabel/wmt22-comet-da")
    return load_from_checkpoint(model_path)
