"""Schema and consistency checks (paper §3.6, Table 1).

Every check below corresponds to a constraint stated in the paper:

    * 5--10 candidate snippets per warning (§3.4, §3.5).
    * Up to two labeling passes per warning, ≤ 3 essentials per warning (§3.6).
    * Every essential label has a non-empty rationale (§3.6, Table 1: 100%).
    * Line spans are well-formed and within the file (§3.6).
    * At least one useful (essential or helpful) label per warning (Table 1: 100%).
    * Project-disjoint splits (§3.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..schema import (
    MAX_ESSENTIAL_PER_WARNING,
    MAX_SNIPPETS_PER_WARNING,
    MIN_SNIPPETS_PER_WARNING,
    SNIPPET_TYPES,
    Warning,
)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked: int = 0

    @property
    def passed(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [
            f"Checked {self.checked} warnings.",
            f"Errors:   {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
        ]
        for e in self.errors[:10]:
            lines.append(f"  ERR  {e}")
        if len(self.errors) > 10:
            lines.append(f"  ... and {len(self.errors) - 10} more errors")
        return "\n".join(lines)


def validate_warning(
    w: Warning,
    report: ValidationReport,
    *,
    require_labels: bool = True,
) -> None:
    """Run every per-warning constraint check."""
    wid = w.warning_id

    # ---- snippet count -------------------------------------------------
    n_snippets = len(w.candidate_snippets)
    if n_snippets < MIN_SNIPPETS_PER_WARNING or n_snippets > MAX_SNIPPETS_PER_WARNING:
        report.errors.append(
            f"{wid}: candidate_snippets count {n_snippets} outside "
            f"[{MIN_SNIPPETS_PER_WARNING}, {MAX_SNIPPETS_PER_WARNING}]"
        )

    # ---- snippet ids unique --------------------------------------------
    snippet_ids = [s.snippet_id for s in w.candidate_snippets]
    if len(snippet_ids) != len(set(snippet_ids)):
        report.errors.append(f"{wid}: duplicate snippet_id values")

    # ---- snippet types valid -------------------------------------------
    for s in w.candidate_snippets:
        if s.type not in SNIPPET_TYPES:
            report.errors.append(f"{wid}/{s.snippet_id}: invalid type {s.type!r}")
        if s.line_start < 1 or s.line_end < s.line_start:
            report.errors.append(
                f"{wid}/{s.snippet_id}: malformed line span "
                f"[{s.line_start}, {s.line_end}]"
            )
        if s.token_count < 0:
            report.errors.append(
                f"{wid}/{s.snippet_id}: negative token_count {s.token_count}"
            )

    # ---- labels reference real snippets --------------------------------
    snippet_set = set(snippet_ids)
    seen_snippets: set[str] = set()
    for lbl in w.labels:
        if lbl.snippet_id not in snippet_set:
            report.errors.append(
                f"{wid}: label references unknown snippet_id {lbl.snippet_id!r}"
            )
        if lbl.snippet_id in seen_snippets:
            report.errors.append(
                f"{wid}: duplicate label for snippet_id {lbl.snippet_id!r}"
            )
        seen_snippets.add(lbl.snippet_id)
        if lbl.annotator_count not in (1, 2):
            report.errors.append(
                f"{wid}/{lbl.snippet_id}: annotator_count must be 1 or 2"
            )
        if lbl.relevance == "essential" and not (lbl.rationale and lbl.rationale.strip()):
            report.errors.append(
                f"{wid}/{lbl.snippet_id}: essential label missing rationale"
            )

    # ---- ≤ 3 essentials per warning ------------------------------------
    n_essential = sum(1 for lbl in w.labels if lbl.relevance == "essential")
    if n_essential > MAX_ESSENTIAL_PER_WARNING:
        report.errors.append(
            f"{wid}: {n_essential} essential labels exceeds limit "
            f"({MAX_ESSENTIAL_PER_WARNING})"
        )

    # ---- at least one useful label -------------------------------------
    if require_labels and not w.useful():
        report.errors.append(f"{wid}: no useful (essential/helpful) labels")


def validate_splits(warnings: Iterable[Warning], report: ValidationReport) -> None:
    """Project-disjoint splits (paper §3.5)."""
    project_split: dict[str, str] = {}
    for w in warnings:
        existing = project_split.get(w.project)
        if existing is None:
            project_split[w.project] = w.split
        elif existing != w.split:
            report.errors.append(
                f"project {w.project!r} appears in splits {existing!r} and "
                f"{w.split!r}"
            )


def validate_dataset(
    warnings: Iterable[Warning],
    *,
    require_labels: bool = True,
) -> ValidationReport:
    report = ValidationReport()
    materialized = list(warnings)
    report.checked = len(materialized)
    for w in materialized:
        validate_warning(w, report, require_labels=require_labels)
    validate_splits(materialized, report)
    return report
