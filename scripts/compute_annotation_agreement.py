"""Recompute annotation agreement from raw per-pass labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from saw_bench_cs.construction.annotation import cohens_kappa, load_passes
from saw_bench_cs.io import load_warnings


LABELS = ("essential", "helpful", "irrelevant")


def pabak(a: list[str], b: list[str], n_classes: int = 3) -> float:
    """Prevalence-Adjusted Bias-Adjusted Kappa.

    For ``n_classes == 2`` this reduces to ``2 * p_o - 1`` (Byrt et al. 1993).
    For the 3-class relevance scheme used here we apply the multi-class
    generalization ``(n * p_o - 1) / (n - 1)`` (Sim & Wright 2005).
    """
    if not a:
        return float("nan")
    po = sum(1 for left, right in zip(a, b) if left == right) / len(a)
    return (n_classes * po - 1) / (n_classes - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/saw_bench_cs.jsonl")
    parser.add_argument("--passes", default="annotation/annotator_passes.jsonl")
    parser.add_argument("--out-json", default="annotation/annotator_agreement.json")
    parser.add_argument("--out-csv", default="annotation/raw_annotation_labels.csv")
    args = parser.parse_args()

    warnings = load_warnings(args.data)
    passes = load_passes(args.passes)
    by_warning: dict[str, list] = {}
    for row in passes:
        by_warning.setdefault(row.warning_id, []).append(row)

    a_labels: list[str] = []
    b_labels: list[str] = []
    raw_rows: list[dict[str, str]] = []
    confusion = Counter()
    for warning in warnings:
        pair = sorted(by_warning.get(warning.warning_id, []), key=lambda p: p.annotator)
        if len(pair) != 2:
            raise SystemExit(f"{warning.warning_id}: expected exactly two passes")
        left, right = pair
        for snippet in warning.candidate_snippets:
            left_label = left.labels.get(snippet.snippet_id, "irrelevant")
            right_label = right.labels.get(snippet.snippet_id, "irrelevant")
            a_labels.append(left_label)
            b_labels.append(right_label)
            confusion[(left_label, right_label)] += 1
            raw_rows.append({
                "warning_id": warning.warning_id,
                "project": warning.project,
                "split": warning.split,
                "snippet_id": snippet.snippet_id,
                "snippet_type": snippet.type,
                "annotator_A": left_label,
                "annotator_B": right_label,
                "merged_relevance": warning.label_of(snippet.snippet_id),
            })

    label_distribution = Counter()
    rationale_count = 0
    for warning in warnings:
        for label in warning.labels:
            label_distribution[label.relevance] += 1
            if label.relevance == "essential" and label.rationale:
                rationale_count += 1

    matrix = [
        [confusion[(row_label, col_label)] for col_label in LABELS]
        for row_label in LABELS
    ]
    agreement = {
        "description": "Agreement recomputed from annotation/annotator_passes.jsonl.",
        "annotation_protocol": (
            "Two released A/B artifact-labeling passes per warning; the merged "
            "labels make the benchmark artifact self-contained and auditable."
        ),
        "main_labeling": {
            "total_warnings_labeled": len(warnings),
            "total_snippets_labeled": len(a_labels),
            "overall_cohens_kappa": round(cohens_kappa(a_labels, b_labels), 4),
            "prevalence_adjusted_kappa": round(pabak(a_labels, b_labels), 4),
            "confusion_matrix": {
                "rows_annotator_A": list(LABELS),
                "cols_annotator_B": list(LABELS),
                "matrix": matrix,
            },
        },
        "label_distribution": {
            key: label_distribution.get(key, 0)
            for key in LABELS
        },
        "quality_checks": {
            "essential_labels_with_rationale": (
                f"{rationale_count}/{label_distribution.get('essential', 0)}"
            ),
            "records_with_useful_context": sum(1 for warning in warnings if warning.useful()),
        },
        "generated_by": "scripts/compute_annotation_agreement.py",
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(agreement, indent=2), encoding="utf-8")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)

    print(json.dumps(agreement, indent=2))
    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
