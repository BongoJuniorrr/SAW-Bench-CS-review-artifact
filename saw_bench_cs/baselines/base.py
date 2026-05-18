"""Ranker protocol (paper §5.2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..schema import Warning


Ranking = list[str]  # ordered list of snippet_id, best first


@runtime_checkable
class Ranker(Protocol):
    name: str

    def fit(self, warnings: list[Warning]) -> None:  # optional
        ...

    def rank(self, warning: Warning) -> Ranking:
        ...


class _NoopFitMixin:
    def fit(self, warnings):  # type: ignore[override]
        return None
