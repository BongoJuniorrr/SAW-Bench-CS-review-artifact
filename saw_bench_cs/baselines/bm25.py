"""BM25 baseline (paper §5.2).

Per-warning BM25 over the candidate snippets, queried with the warning message,
rule_id, and category. Each warning has its own short corpus (5-10 snippets),
so we instantiate a small BM25Okapi per warning. This avoids leakage across
warnings and matches the paper's default message + rule + category query.
"""

from __future__ import annotations

import re
from typing import Iterable

from ..schema import Warning
from .base import Ranking, _NoopFitMixin


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def _query(w: Warning, fields: Iterable[str]) -> list[str]:
    parts: list[str] = []
    for f in fields:
        v = getattr(w, f, "") or ""
        parts.append(str(v))
    return _tokenize(" ".join(parts))


class BM25Ranker(_NoopFitMixin):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75,
                 query_fields: tuple[str, ...] = ("warning_message", "rule_id", "category")):
        self.k1 = k1
        self.b = b
        self.query_fields = query_fields

    def rank(self, warning: Warning) -> Ranking:
        from rank_bm25 import BM25Okapi

        if not warning.candidate_snippets:
            return []
        docs = [_tokenize(s.text) for s in warning.candidate_snippets]
        # rank_bm25 fails on empty corpora — pad with a single space token.
        for i, d in enumerate(docs):
            if not d:
                docs[i] = [" "]
        bm25 = BM25Okapi(docs, k1=self.k1, b=self.b)
        scores = bm25.get_scores(_query(warning, self.query_fields))
        order = sorted(
            range(len(warning.candidate_snippets)),
            key=lambda i: (-scores[i], i),
        )
        return [warning.candidate_snippets[i].snippet_id for i in order]
