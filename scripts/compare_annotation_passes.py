"""Compare two independent annotation-pass JSONL files.

This script is intentionally separate from ``compute_annotation_agreement.py``:
the latter recomputes agreement for the benchmark's released label passes,
whereas this script supports external audits that should not automatically
replace the benchmark labels.
"""

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
    if not a:
        return float("nan")
    po = sum(1 for left, right in zip(a, b) if left == right) / len(a)
    return (n_classes * po - 1) / (n_classes - 1)


def binary_kappa(a: list[str], b: list[str]) -> float:
    a_binary = ["useful" if label != "irrelevant" else "irrelevant" for label in a]
    b_binary = ["useful" if label != "irrelevant" else "irrelevant" for label in b]
    return cohens_kappa(a_binary, b_binary)


def load_single_pass(path: str) -> dict[str, object]:
    rows = load_passes(path)
    out = {}
    for row in rows:
        if row.warning_id in out:
            raise SystemExit(f"{path}: duplicate warning_id {row.warning_id}")
        out[row.warning_id] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/saw_bench_cs.jsonl")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    warnings = load_warnings(args.data)
    left = load_single_pass(args.left)
    right = load_single_pass(args.right)

    left_labels: list[str] = []
    right_labels: list[str] = []
    rows: list[dict[str, str]] = []
    confusion = Counter()
    missing = {"left_warnings": [], "right_warnings": [], "left_snippets": 0, "right_snippets": 0}

    for warning in warnings:
        left_pass = left.get(warning.warning_id)
        right_pass = right.get(warning.warning_id)
        if left_pass is None:
            missing["left_warnings"].append(warning.warning_id)
            continue
        if right_pass is None:
            missing["right_warnings"].append(warning.warning_id)
            continue

        for snippet in warning.candidate_snippets:
            if snippet.snippet_id not in left_pass.labels:
                missing["left_snippets"] += 1
            if snippet.snippet_id not in right_pass.labels:
                missing["right_snippets"] += 1
            left_label = left_pass.labels.get(snippet.snippet_id, "irrelevant")
            right_label = right_pass.labels.get(snippet.snippet_id, "irrelevant")
            left_labels.append(left_label)
            right_labels.append(right_label)
            confusion[(left_label, right_label)] += 1
            rows.append({
                "warning_id": warning.warning_id,
                "project": warning.project,
                "split": warning.split,
                "snippet_id": snippet.snippet_id,
                "snippet_type": snippet.type,
                args.left_name: left_label,
                args.right_name: right_label,
                "benchmark_relevance": warning.label_of(snippet.snippet_id),
            })

    matrix = [[confusion[(row_label, col_label)] for col_label in LABELS] for row_label in LABELS]
    total = len(left_labels)
    observed = sum(1 for left_label, right_label in zip(left_labels, right_labels) if left_label == right_label) / total
    left_counts = Counter(left_labels)
    right_counts = Counter(right_labels)
    useful_left = sum(count for label, count in left_counts.items() if label != "irrelevant")
    useful_right = sum(count for label, count in right_counts.items() if label != "irrelevant")

    summary = {
        "description": "Independent annotation-pass audit; not used to construct data/saw_bench_cs.jsonl.",
        "data": args.data,
        "left": {"name": args.left_name, "path": args.left, "label_distribution": dict(left_counts)},
        "right": {"name": args.right_name, "path": args.right, "label_distribution": dict(right_counts)},
        "total_warnings": len(warnings),
        "total_snippets": total,
        "coverage": missing,
        "agreement": {
            "observed_agreement": round(observed, 4),
            "cohens_kappa": round(cohens_kappa(left_labels, right_labels), 4),
            "prevalence_adjusted_kappa": round(pabak(left_labels, right_labels), 4),
            "binary_useful_kappa": round(binary_kappa(left_labels, right_labels), 4),
            "confusion_matrix": {
                "rows": args.left_name,
                "cols": args.right_name,
                "labels": list(LABELS),
                "matrix": matrix,
            },
        },
        "useful_snippet_counts": {
            args.left_name: useful_left,
            args.right_name: useful_right,
        },
        "interpretation": (
            "This audit shows whether an external pass supports the benchmark labels. "
            "Low agreement should be treated as construct-validity evidence, not as a "
            "reason to silently merge labels."
        ),
        "generated_by": "scripts/compare_annotation_passes.py",
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
