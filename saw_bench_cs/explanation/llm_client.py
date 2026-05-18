"""Vendor-agnostic small LLM client used by the reranker and the RQ4 study."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    endpoint_env: str = "SAW_LLM_ENDPOINT"
    model: str = "phi-3-mini-instruct"
    temperature: float = 0.0
    timeout_s: int = 30


class LLMClient:
    """Tiny HTTP wrapper that accepts OpenAI-compatible or Ollama endpoints."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        env_model = os.environ.get("SAW_LLM_MODEL")
        if env_model:
            self.config.model = env_model

    @property
    def endpoint(self) -> Optional[str]:
        return os.environ.get(self.config.endpoint_env)

    def is_configured(self) -> bool:
        return bool(self.endpoint)

    def complete(self, prompt: str) -> Optional[str]:
        if not self.is_configured():
            return None
        try:
            import requests
        except Exception:
            return None
        try:
            r = requests.post(
                self.endpoint,
                json={
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.config.temperature,
                },
                timeout=self.config.timeout_s,
            )
            r.raise_for_status()
            data = r.json()
        except Exception:
            return None
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"].get("content")
        if "message" in data:
            return data["message"].get("content")
        if "response" in data:
            return data["response"]
        return json.dumps(data)
