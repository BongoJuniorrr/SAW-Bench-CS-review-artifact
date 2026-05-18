"""Render result tables matching paper §5 LaTeX."""

from __future__ import annotations

from typing import Iterable

from .runner import EvaluationResult


_TABLE_4_HEADER = (
    "Method", "R@1", "R@3", "R@5", "MRR", "nDCG@5", "Avg. Tokens",
)


def render_table_4(results: Iterable[EvaluationResult]) -> str:
    """Plain-text Table 4 (paper §5.4): context selection on the test split."""
    rows = [_TABLE_4_HEADER]
    for r in results:
        rows.append((
            r.method,
            f"{r.metrics.get('recall@1', 0):.2f}",
            f"{r.metrics.get('recall@3', 0):.2f}",
            f"{r.metrics.get('recall@5', 0):.2f}",
            f"{r.metrics.get('mrr', 0):.2f}",
            f"{r.metrics.get('ndcg@5', 0):.2f}",
            f"{r.metrics.get('avg_tokens@3', 0):.0f}",
        ))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)
