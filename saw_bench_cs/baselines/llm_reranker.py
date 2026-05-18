"""Small-LLM reranker (paper §5.2).

Reranks the top-5 BM25 candidates using a small instruction model. The model
client is vendor-agnostic and gated on an environment variable so the rest of
the pipeline runs without LLM access.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from ..schema import Warning
from .base import Ranking
from .bm25 import BM25Ranker


_RERANK_PROMPT = """\
You are scoring candidate code snippets for usefulness in explaining a static
analysis warning. Return a JSON list of `snippet_id` strings ordered most
useful first. Use only the snippets provided.

Warning ({rule_id}): {message}
Warning line: {line_text}

Candidates:
{candidates}

Return exactly: {{"order": ["s01", ...]}}
"""


class LLMReranker:
    name = "llm_reranker"

    def __init__(
        self,
        rerank_top_k: int = 5,
        endpoint_env: str = "SAW_LLM_ENDPOINT",
        model: str = "phi-3-mini-instruct",
        temperature: float = 0.0,
        bm25: Optional[BM25Ranker] = None,
    ):
        self.rerank_top_k = rerank_top_k
        self.endpoint_env = endpoint_env
        self.model = os.environ.get("SAW_LLM_MODEL", model)
        self.temperature = temperature
        self.bm25 = bm25 or BM25Ranker()

    def fit(self, warnings: list[Warning]) -> None:
        self.bm25.fit(warnings)

    # ------------------------------------------------------------ helpers
    def _endpoint(self) -> Optional[str]:
        return os.environ.get(self.endpoint_env)

    def _call_llm(self, prompt: str) -> Optional[str]:
        endpoint = self._endpoint()
        if not endpoint:
            return None
        try:
            import requests
        except Exception:
            return None
        try:
            r = requests.post(
                endpoint,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                },
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            # Accept either OpenAI-style or Ollama-style responses.
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            if "message" in data:
                return data["message"].get("content", "")
            return str(data)
        except Exception:
            return None

    @staticmethod
    def _parse_order(text: str) -> list[str]:
        # Find first JSON object containing "order".
        match = re.search(r"\{[^}]*\"order\"[^}]*\}", text or "", re.DOTALL)
        if not match:
            return []
        try:
            return list(json.loads(match.group(0))["order"])
        except (ValueError, KeyError, TypeError):
            return []

    # ---------------------------------------------------------------- rank
    def rank(self, warning: Warning) -> Ranking:
        bm25_order = self.bm25.rank(warning)
        if not bm25_order:
            return []
        head, tail = bm25_order[: self.rerank_top_k], bm25_order[self.rerank_top_k:]

        # If the LLM endpoint isn't configured, fall through to BM25.
        if not self._endpoint():
            return bm25_order

        snippet_lookup = {s.snippet_id: s for s in warning.candidate_snippets}
        candidates_text = "\n".join(
            f"{sid} ({snippet_lookup[sid].type}): {snippet_lookup[sid].text[:240]}"
            for sid in head
        )
        prompt = _RERANK_PROMPT.format(
            rule_id=warning.rule_id,
            message=warning.warning_message,
            line_text=warning.warning_context.warning_line,
            candidates=candidates_text,
        )
        response = self._call_llm(prompt)
        if not response:
            return bm25_order
        new_head = self._parse_order(response)
        # Keep only ids that were actually in the top-k.
        head_set = set(head)
        cleaned = [sid for sid in new_head if sid in head_set]
        # Append any head candidates the model omitted, preserving BM25 order.
        for sid in head:
            if sid not in cleaned:
                cleaned.append(sid)
        return cleaned + tail
