"""Compute context-selection metrics by SpotBugs category using saved rankings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from saw_bench_cs.evaluation.metrics import aggregate
from saw_bench_cs.io import load_warnings, warnings_by_split


def _load_rankings(path: Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rankings[row["warning_id"]] = list(row["ranking"])
    return rankings


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/saw_bench_cs.jsonl")
    p.add_argument("--rankings", default="results_reproduced/baselines/rankings")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="results_reproduced/per_category_metrics.csv")
    args = p.parse_args()

    eval_set = warnings_by_split(load_warnings(args.data), args.split)
    by_category: dict[str, list] = {}
    for warning in eval_set:
        by_category.setdefault(warning.category, []).append(warning)

    desired_methods = {
        "local_first",
        "bm25",
        "embedding_all-MiniLM-L6-v2",
        "type_priority",
        "random",
    }

    ranking_paths = sorted(Path(args.rankings).glob("*.jsonl"))
    rows: list[dict] = []

    for ranking_path in ranking_paths:
        method = ranking_path.stem
        if method not in desired_methods:
            continue

        rankings = _load_rankings(ranking_path)
        for category, category_warnings in sorted(by_category.items()):
            metrics = aggregate(category_warnings, rankings)
            rows.append(
                {
                    "category": category,
                    "method": method,
                    "warnings": len(category_warnings),
                    "recall@3": metrics.get("recall@3", float("nan")),
                    "ndcg@5": metrics.get("ndcg@5", float("nan")),
                    "non_local_recovery@3": metrics.get("non_local_recovery@3", float("nan")),
                    "avg_tokens@3": metrics.get("avg_tokens@3", float("nan")),
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category",
        "method",
        "warnings",
        "recall@3",
        "ndcg@5",
        "non_local_recovery@3",
        "avg_tokens@3",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
