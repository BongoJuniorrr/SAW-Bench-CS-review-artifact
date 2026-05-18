"""Local-first ranker (paper §5.2)."""

from __future__ import annotations

from ..schema import Warning
from .base import Ranking, _NoopFitMixin


_LOCAL_ORDER = ("warning_line", "enclosing_method", "enclosing_class")


class LocalFirstRanker(_NoopFitMixin):
    name = "local_first"

    def rank(self, warning: Warning) -> Ranking:
        # local types in fixed order, then everything else by line distance to the warning.
        local: list[str] = []
        for type_name in _LOCAL_ORDER:
            for s in warning.candidate_snippets:
                if s.type == type_name and s.snippet_id not in local:
                    local.append(s.snippet_id)
                    break
        rest = [s for s in warning.candidate_snippets if s.snippet_id not in local]

        def distance_key(s):
            mid = (s.line_start + s.line_end) / 2
            return abs(mid - warning.line)

        rest.sort(key=distance_key)
        return local + [s.snippet_id for s in rest]
