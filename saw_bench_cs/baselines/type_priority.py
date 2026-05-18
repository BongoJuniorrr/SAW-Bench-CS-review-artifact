"""Static type-priority ranker (paper §5.2).

Validation-set priors P(essential | snippet_type) determine the primary
ordering, with line proximity to the warning as a tiebreaker.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

from ..schema import SNIPPET_TYPES, SnippetType, Warning
from .base import Ranking


# Default priors derived from paper Table 2 (% essential per type).
# Used when fit() is not called or no validation labels are available.
_DEFAULT_PRIORS: dict[SnippetType, float] = {
    "warning_line": 0.61,
    "enclosing_method": 0.546,
    "enclosing_class": 0.181,
    "caller": 0.178,
    "callee": 0.216,
    "field_or_type_declaration": 0.291,
    "annotation_or_contract": 0.335,
    "test": 0.112,
    "configuration": 0.086,
    "similar_code": 0.093,
    "history_or_diff": 0.12,
}


def fit_priors(warnings: Iterable[Warning]) -> dict[SnippetType, float]:
    """P(essential | type) computed over the supplied warnings.

    Add-one smoothing keeps unseen types from collapsing to zero.
    """
    counts: dict[SnippetType, int] = defaultdict(int)
    essentials: dict[SnippetType, int] = defaultdict(int)
    for w in warnings:
        labels = {lbl.snippet_id: lbl.relevance for lbl in w.labels}
        for s in w.candidate_snippets:
            counts[s.type] += 1
            if labels.get(s.snippet_id) == "essential":
                essentials[s.type] += 1
    priors: dict[SnippetType, float] = {}
    for t in SNIPPET_TYPES:
        n = counts.get(t, 0)
        e = essentials.get(t, 0)
        priors[t] = (e + 1) / (n + 2)  # Laplace smoothing
    return priors


class TypePriorityRanker:
    name = "type_priority"

    def __init__(
        self,
        priors: Optional[dict[SnippetType, float]] = None,
        proximity_weight: float = 0.3,
    ):
        self.priors = dict(priors) if priors else dict(_DEFAULT_PRIORS)
        self.proximity_weight = proximity_weight

    def fit(self, warnings: list[Warning]) -> None:
        if warnings:
            self.priors = fit_priors(warnings)

    def rank(self, warning: Warning) -> Ranking:
        def score(s):
            base = self.priors.get(s.type, 0.0)
            mid = (s.line_start + s.line_end) / 2
            distance = abs(mid - warning.line) if warning.line > 0 else 0
            proximity = 1.0 / (1.0 + distance / 50.0)  # decays slowly
            return base + self.proximity_weight * proximity

        ordered = sorted(
            warning.candidate_snippets,
            key=lambda s: (-score(s), s.line_start, s.snippet_id),
        )
        return [s.snippet_id for s in ordered]
