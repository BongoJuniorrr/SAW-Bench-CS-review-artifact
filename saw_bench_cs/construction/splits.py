"""Project-based train/validation/test splits (paper §3.5, §4.5)."""

from __future__ import annotations

from pathlib import Path

from ..schema import Warning


def load_split_assignment(path: str | Path) -> dict[str, str]:
    """Load configs/splits.yaml → {project_name: split}."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assignment: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        for project in raw.get(split, []):
            assignment[project] = split
    return assignment


def apply_splits(warnings: list[Warning], assignment: dict[str, str]) -> list[Warning]:
    """Set Warning.split based on the project assignment, mutating in place."""
    out = []
    for w in warnings:
        split = assignment.get(w.project)
        if split is None:
            raise ValueError(
                f"warning {w.warning_id}: project {w.project!r} not in split assignment"
            )
        w.split = split
        out.append(w)
    return out


def assert_disjoint(warnings: list[Warning]) -> None:
    """Verify that no project appears in more than one split."""
    seen: dict[str, str] = {}
    for w in warnings:
        existing = seen.get(w.project)
        if existing is None:
            seen[w.project] = w.split
        elif existing != w.split:
            raise ValueError(
                f"project {w.project!r} appears in both {existing!r} and {w.split!r}"
            )
