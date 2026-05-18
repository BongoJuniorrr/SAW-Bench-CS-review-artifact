"""Token counting utility used for snippet token_count and avg-tokens metrics.

Prefers tiktoken's cl100k_base; falls back to a deterministic whitespace +
punctuation tokenizer when tiktoken is unavailable, so the metrics module is
usable in offline test environments.
"""

from __future__ import annotations

import functools
import re
from typing import Iterable


_FALLBACK_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")


@functools.lru_cache(maxsize=1)
def _tiktoken_encoder():
    try:
        import tiktoken  # type: ignore

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - tiktoken not installed
        return None


def count_tokens(text: str) -> int:
    """Count tokens in `text`. Uses tiktoken when available, else fallback."""
    if not text:
        return 0
    enc = _tiktoken_encoder()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return len(_FALLBACK_PATTERN.findall(text))


def total_tokens(snippets: Iterable) -> int:
    return sum(getattr(s, "token_count", 0) or count_tokens(getattr(s, "text", ""))
               for s in snippets)
