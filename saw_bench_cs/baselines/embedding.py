"""Embedding baseline (paper §5.2).

Cosine similarity between (warning_message + rule_id + category) and each
candidate snippet text, using a compact local model
(`sentence-transformers/all-MiniLM-L6-v2` by default). Falls back to a
hashed bag-of-words embedding when sentence-transformers is unavailable so
the baseline is still runnable in offline test environments.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

from ..schema import Warning
from .base import Ranking, _NoopFitMixin


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def _hash_embedding(text: str, dim: int = 256) -> list[float]:
    """Deterministic hashed bag-of-words embedding (fallback)."""
    vec = [0.0] * dim
    for tok in _TOKEN.findall(text or ""):
        h = int(hashlib.md5(tok.lower().encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vec[idx] += sign
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # both unit-normalized


class EmbeddingRanker(_NoopFitMixin):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True,
        allow_hash_fallback: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.allow_hash_fallback = allow_hash_fallback
        self._model: Optional[object] = None
        self._resolved_name: Optional[str] = None

    @staticmethod
    def _label_for_model(model_name: str) -> str:
        model_id = model_name.split("/")[-1]
        return f"embedding_{model_id}"

    def _resolve_name(self) -> str:
        if self._resolved_name is not None:
            return self._resolved_name
        if self._model is False:
            self._resolved_name = "embedding_hashed_fallback"
            return self._resolved_name
        model = self._ensure_model()
        if model:
            self._resolved_name = self._label_for_model(self.model_name)
            return self._resolved_name
        if not self.allow_hash_fallback:
            raise RuntimeError(
                "sentence-transformers unavailable and hash fallback is disabled"
            )
        self._resolved_name = "embedding_hashed_fallback"
        return self._resolved_name

    @property
    def name(self) -> str:
        return self._resolve_name()

    def _ensure_model(self):
        if self._model is not None or self._model is False:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_name, device=self.device)
        except Exception:
            if not self.allow_hash_fallback:
                raise
            self._model = False  # signal: use fallback
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        if model:
            import numpy as np  # type: ignore

            vecs = model.encode(texts, normalize_embeddings=self.normalize)
            return [list(map(float, v)) for v in vecs]
        return [_hash_embedding(t) for t in texts]

    def rank(self, warning: Warning) -> Ranking:
        if not warning.candidate_snippets:
            return []
        query = " ".join([
            warning.warning_message or "",
            warning.rule_id or "",
            warning.category or "",
        ])
        texts = [query] + [s.text for s in warning.candidate_snippets]
        embeddings = self._encode(texts)
        q_vec = embeddings[0]
        scores = [_cosine(q_vec, v) for v in embeddings[1:]]
        order = sorted(
            range(len(warning.candidate_snippets)),
            key=lambda i: (-scores[i], i),
        )
        return [warning.candidate_snippets[i].snippet_id for i in order]
