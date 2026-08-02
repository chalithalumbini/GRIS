# embedder.py
from __future__ import annotations

from typing import List, Dict, Optional
import torch
import numpy as np


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = "glove-wiki-gigaword-300",
        embedding_type: str = "word",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.embedding_type = embedding_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.word_vectors = None
        self.transformer_model = None
        self.embedding_dim = None

        if embedding_type == "word":
            self._load_word_embeddings()
            self.model = None
        elif embedding_type == "transformer":
            self._load_transformer()
            self.model = self.transformer_model
        else:
            raise ValueError("embedding_type must be 'word' or 'transformer'")

    def _load_word_embeddings(self):
        import os
        import gensim.downloader as api
        from gensim.models import KeyedVectors

        # gensim.downloader.load() already caches the RAW download on disk
        # (default: ~/gensim-data) and won't re-fetch it over the network.
        # But it still re-parses that raw file (a large plain-text vector
        # file) from scratch on every call, which is slow and looks just
        # like a fresh download. To fix that, cache the *parsed* vectors in
        # gensim's fast binary format (mmap-loadable) after the first load,
        # so every run after the first is near-instant instead of a full
        # re-parse.
        cache_dir = os.environ.get(
            "GRIS_EMBED_CACHE_DIR",
            os.path.join(os.path.expanduser("~"), ".gris_cache", "word_vectors"),
        )
        os.makedirs(cache_dir, exist_ok=True)
        fast_path = os.path.join(cache_dir, f"{self.model_name}.kv")

        if os.path.exists(fast_path):
            print(f"[INFO] Loading cached word embeddings from {fast_path} "
                  f"(fast binary format)...")
            self.word_vectors = KeyedVectors.load(fast_path, mmap="r")
        else:
            print(f"[INFO] Loading word embeddings: {self.model_name} "
                  f"(first run — this is slow, but only happens once)...")
            self.word_vectors = api.load(self.model_name)
            print(f"[INFO] Caching parsed vectors to {fast_path} for fast reuse "
                  f"on future runs...")
            self.word_vectors.save(fast_path)

        self.embedding_dim = int(self.word_vectors.vector_size)
        print(f"[OK] Loaded {len(self.word_vectors)} word vectors (dim={self.embedding_dim})")

    def _load_transformer(self):
        from sentence_transformers import SentenceTransformer
        print(f"[INFO] Loading sentence transformer: {self.model_name}...")
        self.transformer_model = SentenceTransformer(self.model_name, device=self.device)
        test = self.transformer_model.encode("test", convert_to_tensor=True)
        self.embedding_dim = int(test.shape[-1])
        print(f"[OK] Loaded transformer model (dim={self.embedding_dim})")

    def get_word_embedding(self, word: str) -> torch.Tensor:
        if self.embedding_type != "word":
            raise ValueError("get_word_embedding() only for embedding_type='word'")

        if not word:
            return torch.zeros(self.embedding_dim, dtype=torch.float32, device=self.device)

        if word in self.word_vectors:
            return torch.tensor(self.word_vectors[word], dtype=torch.float32, device=self.device)

        w = word.lower()
        if w in self.word_vectors:
            return torch.tensor(self.word_vectors[w], dtype=torch.float32, device=self.device)

        return torch.zeros(self.embedding_dim, dtype=torch.float32, device=self.device)

    def encode(self, text: str) -> np.ndarray:
        if self.embedding_type == "transformer":
            return self.transformer_model.encode(text, convert_to_numpy=True)
        else:
            words = text.split()
            if not words:
                return np.zeros(self.embedding_dim, dtype=np.float32)
            vecs = [self.get_word_embedding(w).detach().cpu().numpy() for w in words]
            return np.mean(vecs, axis=0) if vecs else np.zeros(self.embedding_dim, dtype=np.float32)

    def compute_similarity(self, text1: str, text2: str) -> float:
        a = self.encode(text1)
        b = self.encode(text2)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))

    def similarity(self, text1: str, text2: str) -> float:
        return self.compute_similarity(text1, text2)

    def embed_tokens(self, dep_sentence) -> Dict[int, torch.Tensor]:
        if self.embedding_type == "word":
            return {t.id: self.get_word_embedding(t.text) for t in dep_sentence.tokens}
        return self._embed_tokens_transformer(dep_sentence)

    def _get_hf_modules(self):
        """
        Access the underlying HF tokenizer + model inside SentenceTransformer.
        Works across common SentenceTransformer versions.
        """
        try:
            mod = self.transformer_model._first_module()
            tokenizer = getattr(mod, "tokenizer", None)
            model = getattr(mod, "auto_model", None)
            if tokenizer is not None and model is not None:
                return tokenizer, model
        except Exception:
            pass

        try:
            mod = self.transformer_model[0]
            tokenizer = getattr(mod, "tokenizer", None)
            model = getattr(mod, "auto_model", None)
            if tokenizer is not None and model is not None:
                return tokenizer, model
        except Exception:
            pass

        raise RuntimeError("Could not access tokenizer/auto_model from SentenceTransformer.")

    # Number of transformer layers to pool for token embeddings.
    # Peters et al. (2018) showed that averaging the last N layers gives
    # richer representations than the final layer alone:
    #   - Lower layers: morphological/syntactic features
    #   - Upper layers: semantic/contextual features
    # For morphologically rich languages (RU, DE, FI), pooling captures
    # both inflectional patterns and word meaning simultaneously.
    # Set to 1 to revert to last-hidden-state only (original behaviour).
    N_LAYERS_POOL = 4

    def _embed_tokens_transformer(self, dep_sentence) -> Dict[int, torch.Tensor]:
        """
        Contextual token embeddings via last-N-layer average pooling.

        Improvement over single last_hidden_state:
          Averaging the last 4 transformer layers captures both syntactic
          (lower layers) and semantic (upper layers) information. This is
          especially beneficial for morphologically rich languages like
          Russian and German where inflected forms need both morphological
          disambiguation and semantic similarity.

        Implementation:
          Requests output_hidden_states=True, takes the last N_LAYERS_POOL
          hidden states, averages them at the subword level, then pools
          subwords to word level by averaging.
        """
        tokenizer, model = self._get_hf_modules()

        words = [t.text for t in dep_sentence.tokens]
        if not words:
            return {}

        encoded = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # IMPORTANT: read word_ids BEFORE losing BatchEncoding methods
        try:
            word_ids = encoded.word_ids(batch_index=0)
        except Exception:
            word_ids = None

        encoded = encoded.to(self.device)
        model = model.to(self.device)
        model.eval()

        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)

        # Pool last N_LAYERS_POOL hidden states
        # hidden_states: tuple of (n_layers+1) tensors each [1, L, D]
        hidden_states = outputs.hidden_states  # includes embedding layer at [0]
        if hidden_states is not None and len(hidden_states) > self.N_LAYERS_POOL:
            # Take last N layers (exclude embedding layer at index 0)
            layers_to_pool = hidden_states[-self.N_LAYERS_POOL:]   # last 4 layers
            # Stack and average: [N, 1, L, D] -> [1, L, D] -> [L, D]
            out = torch.stack(layers_to_pool, dim=0).mean(dim=0)[0]
        else:
            # Fallback: use last_hidden_state if hidden_states unavailable
            out = outputs.last_hidden_state[0]  # [L, D]

        # If word_ids mapping is unavailable, fallback to sentence embedding
        if word_ids is None:
            sent_vec = self.transformer_model.encode(
                " ".join(words), convert_to_tensor=True
            ).to(self.device)
            return {t.id: sent_vec for t in dep_sentence.tokens}

        # Accumulate subword vectors per word
        bucket: Dict[int, List[torch.Tensor]] = {}
        for i, wid in enumerate(word_ids):
            if wid is None:
                continue
            bucket.setdefault(int(wid), []).append(out[i])

        token_embeddings: Dict[int, torch.Tensor] = {}
        for wid, vecs in bucket.items():
            if 0 <= wid < len(dep_sentence.tokens):
                tok_id = dep_sentence.tokens[wid].id
                token_embeddings[tok_id] = torch.stack(vecs).mean(dim=0)

        # Ensure every token has an embedding (zero fallback for unparsed tokens)
        for t in dep_sentence.tokens:
            if t.id not in token_embeddings:
                token_embeddings[t.id] = torch.zeros(
                    self.embedding_dim, device=self.device
                )

        return token_embeddings

    def embed_batch(self, dep_sentences: list) -> List[Dict[int, torch.Tensor]]:
        return [self.embed_tokens(s) for s in dep_sentences]