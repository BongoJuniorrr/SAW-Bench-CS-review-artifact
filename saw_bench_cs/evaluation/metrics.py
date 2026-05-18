"""Ranking metrics used in paper §5.3 and Table 3 diagnostics.

Conventions:
    * `ranking` is a list of snippet_ids ordered best-first.
    * `essentials` / `helpfuls` are sets of snippet_ids.
    * `labels` is the full {snippet_id: relevance} map for the warning.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

from ..schema import (
    LOCAL_SNIPPET_TYPES,
    RELEVANCE_GAIN,
    CandidateSnippet,
    Relevance,
    Warning,
)


# ----------------------------- main metrics ---------------------------------

def recall_at_k(ranking: list[str], essentials: set[str], k: int) -> float:
    """Fraction of essential snippets present in the top-k positions."""
    if not essentials:
        return float("nan")
    top = set(ranking[:k])
    return len(top & essentials) / len(essentials)


def mrr(ranking: list[str], essentials: set[str]) -> float:
    """Mean reciprocal rank of the first essential snippet (1-indexed).

    Returns 0 if no essential is present anywhere in the ranking. Returns NaN
    when there are no essentials at all (so callers can skip the warning).
    """
    if not essentials:
        return float("nan")
    for i, sid in enumerate(ranking, start=1):
        if sid in essentials:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranking: list[str], labels: Mapping[str, Relevance], k: int) -> float:
    """nDCG@k with graded gain (essential=2, helpful=1, irrelevant=0)."""
    if not ranking:
        return 0.0
    gains = [RELEVANCE_GAIN.get(labels.get(sid, "irrelevant"), 0) for sid in ranking[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))

    # Ideal: best possible top-k by descending gain across all labels.
    all_gains = sorted(
        (RELEVANCE_GAIN.get(r, 0) for r in labels.values()),
        reverse=True,
    )[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(all_gains))
    if idcg == 0:
        return 0.0
    return dcg / idcg


# ----------------------------- diagnostics (Table 3) -------------------------

def coverage_at_k(ranking: list[str], useful: set[str], k: int) -> float:
    """Evidence coverage@k: useful snippets in top-k / total useful snippets."""
    if not useful:
        return float("nan")
    return len(set(ranking[:k]) & useful) / len(useful)


def token_normalized_utility(
    ranking: list[str],
    labels: Mapping[str, Relevance],
    snippets: Iterable[CandidateSnippet],
    k: int,
) -> float:
    """Graded relevance gain per 1K selected tokens for the top-k snippets."""
    by_id = {s.snippet_id: s for s in snippets}
    gain = 0
    tokens = 0
    for sid in ranking[:k]:
        gain += RELEVANCE_GAIN.get(labels.get(sid, "irrelevant"), 0)
        s = by_id.get(sid)
        if s is not None:
            tokens += s.token_count
    if tokens == 0:
        return 0.0
    return gain / (tokens / 1000.0)


def distractor_rate_at_k(ranking: list[str], labels: Mapping[str, Relevance], k: int = 3) -> float:
    """Irrelevant snippets in top-k / k."""
    if k <= 0:
        return 0.0
    top = ranking[:k]
    if not top:
        return 0.0
    irrelevant = sum(
        1 for sid in top if labels.get(sid, "irrelevant") == "irrelevant"
    )
    return irrelevant / k


def non_local_recovery_at_k(
    ranking: list[str],
    labels: Mapping[str, Relevance],
    snippets: Iterable[CandidateSnippet],
    k: int = 3,
) -> float:
    """Non-local essential snippets in top-k / total non-local essential snippets.

    Non-local types exclude `warning_line` and `enclosing_method` (paper §5.3).
    """
    by_id = {s.snippet_id: s for s in snippets}
    nonlocal_essentials = {
        sid for sid, rel in labels.items()
        if rel == "essential"
        and by_id.get(sid) is not None
        and by_id[sid].type not in LOCAL_SNIPPET_TYPES
    }
    if not nonlocal_essentials:
        return float("nan")
    return len(set(ranking[:k]) & nonlocal_essentials) / len(nonlocal_essentials)


# --------------------------- token accounting -------------------------------

def average_tokens(rankings: list[tuple[Warning, list[str]]], k: int = 3) -> float:
    """Average top-k snippet token count across (warning, ranking) pairs."""
    totals = []
    for w, r in rankings:
        by_id = {s.snippet_id: s for s in w.candidate_snippets}
        totals.append(sum(
            by_id[sid].token_count
            for sid in r[:k]
            if sid in by_id
        ))
    if not totals:
        return 0.0
    return sum(totals) / len(totals)


def full_context_tokens(warning: Warning) -> int:
    return sum(s.token_count for s in warning.candidate_snippets)


def local_only_tokens(warning: Warning) -> int:
    return sum(
        s.token_count
        for s in warning.candidate_snippets
        if s.type in LOCAL_SNIPPET_TYPES
    )


# --------------------------- aggregation helpers ----------------------------

def _safe_mean(values: list[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def aggregate(
    warnings: list[Warning],
    rankings: dict[str, list[str]],
    *,
    recall_k: tuple[int, ...] = (1, 3, 5),
    ndcg_k: int = 5,
    coverage_k: tuple[int, ...] = (1, 3, 5, 7),
    distractor_k: int = 3,
    non_local_k: int = 3,
) -> dict[str, float]:
    """Aggregate every metric across a list of warnings."""
    results: dict[str, list[float]] = {}

    for w in warnings:
        ranking = rankings.get(w.warning_id, [])
        labels = {lbl.snippet_id: lbl.relevance for lbl in w.labels}
        essentials = w.essentials()
        useful = w.useful()

        for k in recall_k:
            results.setdefault(f"recall@{k}", []).append(recall_at_k(ranking, essentials, k))
        results.setdefault("mrr", []).append(mrr(ranking, essentials))
        results.setdefault(f"ndcg@{ndcg_k}", []).append(ndcg_at_k(ranking, labels, ndcg_k))

        for k in coverage_k:
            results.setdefault(f"coverage@{k}", []).append(coverage_at_k(ranking, useful, k))
        results.setdefault(f"distractor@{distractor_k}", []).append(
            distractor_rate_at_k(ranking, labels, distractor_k)
        )
        results.setdefault(f"non_local_recovery@{non_local_k}", []).append(
            non_local_recovery_at_k(ranking, labels, w.candidate_snippets, non_local_k)
        )
        results.setdefault("token_norm_utility@3", []).append(
            token_normalized_utility(ranking, labels, w.candidate_snippets, 3)
        )

    aggregated = {key: _safe_mean(values) for key, values in results.items()}

    # Token strategy summary (paper Table 5 / §5.5).
    aggregated["avg_tokens@3"] = average_tokens(
        [(w, rankings.get(w.warning_id, [])) for w in warnings], k=3,
    )
    aggregated["avg_tokens_full_context"] = _safe_mean(
        [float(full_context_tokens(w)) for w in warnings]
    )
    aggregated["avg_tokens_local_only"] = _safe_mean(
        [float(local_only_tokens(w)) for w in warnings]
    )
    return aggregated
