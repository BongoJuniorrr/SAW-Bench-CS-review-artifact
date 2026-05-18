"""Evaluate retrieval metrics under alternative merge rules for annotation passes.

This script compares the released benchmark labels against several synthetic
merge policies derived from the underlying A/B annotation passes.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
from saw_bench_cs.construction.annotation import load_passes
from saw_bench_cs.evaluation.metrics import (
    mrr,
    ndcg_at_k,
    non_local_recovery_at_k,
    recall_at_k,
)
from saw_bench_cs.io import load_warnings, warnings_by_split


LABEL_ORDER = {"essential": 2, "helpful": 1, "irrelevant": 0}
VARIANTS = (
    "current_merge",
    "A_only",
    "B_only",
    "strict_agreement_essential",
    "lower_bound_disagreement",
    "useful_binary",
)


def _load_rankings(path: Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rankings[row["warning_id"]] = list(row["ranking"])
    return rankings


def _labels_from_pass(ap) -> dict[str, str]:
    return dict(ap.labels)


def _merge_labels(variant: str, warning, pass_a, pass_b) -> dict[str, str]:
    if variant == "current_merge":
        return {lbl.snippet_id: lbl.relevance for lbl in warning.labels}
    if variant == "A_only":
        return _labels_from_pass(pass_a)
    if variant == "B_only":
        return _labels_from_pass(pass_b)

    labels: dict[str, str] = {}
    snippet_ids = [s.snippet_id for s in warning.candidate_snippets]
    for sid in snippet_ids:
        left = pass_a.labels.get(sid, "irrelevant")
        right = pass_b.labels.get(sid, "irrelevant")
        if variant == "strict_agreement_essential":
            if left == right == "essential":
                labels[sid] = "essential"
            elif left == right == "irrelevant":
                labels[sid] = "irrelevant"
            else:
                labels[sid] = "helpful"
        elif variant == "lower_bound_disagreement":
            labels[sid] = min((left, right), key=lambda label: LABEL_ORDER.get(label, 0))
        elif variant == "useful_binary":
            labels[sid] = "essential" if left != "irrelevant" or right != "irrelevant" else "irrelevant"
        else:
            raise ValueError(f"unknown merge variant {variant!r}")
    return labels


def _evaluate_warning(ranking: list[str], labels: dict[str, str], warning) -> dict[str, float]:
    essentials = {sid for sid, rel in labels.items() if rel == "essential"}
    useful = {sid for sid, rel in labels.items() if rel in {"essential", "helpful"}}
    label_map = dict(labels)
    return {
        "recall@1": recall_at_k(ranking, essentials, 1),
        "recall@3": recall_at_k(ranking, essentials, 3),
        "recall@5": recall_at_k(ranking, essentials, 5),
        "mrr": mrr(ranking, essentials),
        "ndcg@5": ndcg_at_k(ranking, label_map, 5),
        "non_local_recovery@3": non_local_recovery_at_k(ranking, label_map, warning.candidate_snippets, 3),
        "warnings_with_essential": 1.0 if essentials else 0.0,
        "essential_labels": float(len(essentials)),
    }


def _mean(values: list[float]) -> float:
    values = [v for v in values if v == v]
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/saw_bench_cs.jsonl")
    parser.add_argument("--passes", default="annotation/annotator_passes.jsonl")
    parser.add_argument("--rankings", default="results_reproduced/baselines/rankings")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="results_reproduced/merge_sensitivity.csv")
    args = parser.parse_args()

    warnings = warnings_by_split(load_warnings(args.data), args.split)
    pass_rows = load_passes(args.passes)
    by_warning: dict[str, list] = defaultdict(list)
    for row in pass_rows:
        by_warning[row.warning_id].append(row)

    rankings_dir = Path(args.rankings)
    ranking_files = sorted(rankings_dir.glob("*.jsonl"))
    rows: list[dict[str, str]] = []
    for ranking_file in ranking_files:
        rankings = _load_rankings(ranking_file)
        method = ranking_file.stem
        for variant in VARIANTS:
            metric_store: dict[str, list[float]] = defaultdict(list)
            total_warnings_with_essential = 0.0
            total_essential_labels = 0.0
            for warning in warnings:
                pair = sorted(by_warning.get(warning.warning_id, []), key=lambda p: p.annotator)
                if len(pair) != 2:
                    raise SystemExit(f"{warning.warning_id}: expected exactly two annotation passes")
                labels = _merge_labels(variant, warning, pair[0], pair[1])
                metrics = _evaluate_warning(rankings.get(warning.warning_id, []), labels, warning)
                for key, value in metrics.items():
                    metric_store[key].append(value)
                total_warnings_with_essential += metrics["warnings_with_essential"]
                total_essential_labels += metrics["essential_labels"]
            rows.append({
                "variant": variant,
                "method": method,
                "recall@1": f"{_mean(metric_store['recall@1']):.4f}",
                "recall@3": f"{_mean(metric_store['recall@3']):.4f}",
                "recall@5": f"{_mean(metric_store['recall@5']):.4f}",
                "mrr": f"{_mean(metric_store['mrr']):.4f}",
                "ndcg@5": f"{_mean(metric_store['ndcg@5']):.4f}",
                "non_local_recovery@3": f"{_mean(metric_store['non_local_recovery@3']):.4f}",
                "warnings_with_essential": f"{total_warnings_with_essential:.0f}",
                "essential_labels": f"{total_essential_labels:.0f}",
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {out}")


if __name__ == "__main__":
    main()