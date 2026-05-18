"""Label-pass tooling and agreement metrics (paper §3.6).

Produces RelevanceLabel records per snippet from up to two labeling passes and
computes Cohen's κ for audit and reproducibility checks.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..schema import (
    MAX_ESSENTIAL_PER_WARNING,
    CandidateSnippet,
    RelevanceLabel,
    Relevance,
)


@dataclass
class AnnotatorPass:
    """A single labeling pass for one warning."""

    warning_id: str
    annotator: str
    labels: dict[str, Relevance]  # snippet_id -> relevance
    rationales: dict[str, str]    # snippet_id -> rationale (essential only)


def merge_passes(
    passes: list[AnnotatorPass],
) -> list[RelevanceLabel]:
    """Combine two labeling passes into the dataset's RelevanceLabel records.

    Rules:
      * If both passes agree, annotator_count = 2 with their relevance.
      * If they disagree, take the *more useful* label (essential > helpful >
        irrelevant) with annotator_count = 1, preserving the rationale of the
        pass whose label was kept.
    """
    if not passes:
        return []
    if len(passes) > 2:
        raise ValueError("merge_passes expects at most two labeling passes")

    snippet_ids: set[str] = set()
    for p in passes:
        snippet_ids.update(p.labels)

    severity_order = {"essential": 2, "helpful": 1, "irrelevant": 0}
    merged: list[RelevanceLabel] = []
    for sid in sorted(snippet_ids):
        relevances = [p.labels.get(sid, "irrelevant") for p in passes]
        if len(set(relevances)) == 1:
            chosen: Relevance = relevances[0]
            count = len(passes)
            rationale = next(
                (p.rationales.get(sid) for p in passes if p.rationales.get(sid)),
                None,
            )
        else:
            # Disagreement: keep the more useful label.
            best_idx = max(range(len(passes)), key=lambda i: severity_order.get(relevances[i], 0))
            chosen = relevances[best_idx]
            count = 1
            rationale = passes[best_idx].rationales.get(sid)
        if chosen != "essential":
            rationale = None  # rationale only meaningful for essentials
        merged.append(RelevanceLabel(
            snippet_id=sid,
            relevance=chosen,
            annotator_count=count,
            rationale=rationale,
        ))
    return merged


def cohens_kappa(a: list[Relevance], b: list[Relevance]) -> float:
    """Compute Cohen's κ between two equal-length label lists.

    Returns NaN if labels collapse to a single class (κ undefined). Used in the
    30-warning calibration round (paper §3.6).
    """
    if len(a) != len(b):
        raise ValueError("kappa: label lists must be equal length")
    n = len(a)
    if n == 0:
        return float("nan")
    classes = sorted(set(a) | set(b))
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    a_counts = Counter(a)
    b_counts = Counter(b)
    p_e = sum((a_counts[c] / n) * (b_counts[c] / n) for c in classes)
    if p_e >= 1.0:
        return float("nan")
    return (p_o - p_e) / (1 - p_e)


def calibration_passed(
    pass_a: AnnotatorPass,
    pass_b: AnnotatorPass,
    threshold: float = 0.6,
) -> tuple[bool, float]:
    """Check the κ ≥ threshold gate for a single warning."""
    keys = sorted(set(pass_a.labels) | set(pass_b.labels))
    a = [pass_a.labels.get(k, "irrelevant") for k in keys]
    b = [pass_b.labels.get(k, "irrelevant") for k in keys]
    kappa = cohens_kappa(a, b)
    return (kappa is not None and kappa >= threshold, kappa)


def randomize_snippet_order(
    snippets: list[CandidateSnippet],
    seed: int,
) -> list[CandidateSnippet]:
    """Randomize the snippet display order to control position bias."""
    rng = random.Random(seed)
    out = list(snippets)
    rng.shuffle(out)
    return out


def enforce_essential_cap(labels: list[RelevanceLabel]) -> list[RelevanceLabel]:
    """Defensive: clamp essentials to ≤ 3 by demoting tail to helpful."""
    essentials = [lbl for lbl in labels if lbl.relevance == "essential"]
    if len(essentials) <= MAX_ESSENTIAL_PER_WARNING:
        return labels
    keep = set(id(lbl) for lbl in essentials[:MAX_ESSENTIAL_PER_WARNING])
    out: list[RelevanceLabel] = []
    for lbl in labels:
        if lbl.relevance == "essential" and id(lbl) not in keep:
            out.append(RelevanceLabel(
                snippet_id=lbl.snippet_id,
                relevance="helpful",
                annotator_count=lbl.annotator_count,
                rationale=None,
            ))
        else:
            out.append(lbl)
    return out


# --------------------------- I/O for annotation runs ------------------------

def load_passes(path: str | Path) -> list[AnnotatorPass]:
    """Load AnnotatorPass records from a JSONL file produced by scripts/annotate.py."""
    rows: list[AnnotatorPass] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        rows.append(AnnotatorPass(
            warning_id=raw["warning_id"],
            annotator=raw["annotator"],
            labels=raw.get("labels", {}),
            rationales=raw.get("rationales", {}),
        ))
    return rows


def save_passes(passes: Iterable[AnnotatorPass], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for ap in passes:
            fh.write(json.dumps({
                "warning_id": ap.warning_id,
                "annotator": ap.annotator,
                "labels": ap.labels,
                "rationales": ap.rationales,
            }))
            fh.write("\n")
