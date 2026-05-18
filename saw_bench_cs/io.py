"""JSONL load / save for SAW-Bench-CS warning records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from .schema import Warning


def load_warnings(path: str | Path) -> list[Warning]:
    """Load every warning from a JSONL file."""
    return list(iter_warnings(path))


def iter_warnings(path: str | Path) -> Iterator[Warning]:
    """Yield warnings from a JSONL file one record at a time."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Warning.from_dict(json.loads(line))
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(
                    f"Failed to parse warning record on line {line_no} of {p}: {exc}"
                ) from exc


def save_warnings(warnings: Iterable[Warning], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for w in warnings:
            fh.write(json.dumps(w.to_dict(), ensure_ascii=False))
            fh.write("\n")


def warnings_by_split(warnings: Iterable[Warning], split: str) -> list[Warning]:
    return [w for w in warnings if w.split == split]
