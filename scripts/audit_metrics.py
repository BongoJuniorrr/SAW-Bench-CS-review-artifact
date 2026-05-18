"""Audit ranking metrics to explain recall-vs-MRR behavior per warning and method."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import _bootstrap  # noqa: F401
from saw_bench_cs.evaluation.metrics import mrr, recall_at_k
from saw_bench_cs.io import load_warnings, warnings_by_split


def _load_rankings(path: Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rankings[row["warning_id"]] = list(row["ranking"])
    return rankings


def _first_essential_rank(ranking: list[str], essentials: set[str]) -> int:
    for idx, sid in enumerate(ranking, start=1):
        if sid in essentials:
            return idx
    return 0


def _safe_mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan")
    return sum(clean) / len(clean)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/saw_bench_cs.jsonl")
    p.add_argument("--rankings", default="results_reproduced/baselines/rankings")
    p.add_argument("--split", default="test")
    p.add_argument("--out-warning", default="results_reproduced/metric_sanity_by_warning.csv")
    p.add_argument("--out-summary", default="results_reproduced/metric_sanity_summary.csv")
    args = p.parse_args()

    warnings = warnings_by_split(load_warnings(args.data), args.split)
    by_id = {w.warning_id: w for w in warnings}

    ranking_paths = sorted(Path(args.rankings).glob("*.jsonl"))
    if not ranking_paths:
        raise FileNotFoundError(f"No ranking files found under {args.rankings}")

    warning_rows: list[dict] = []
    summary_rows: list[dict] = []

    for ranking_path in ranking_paths:
        method = ranking_path.stem
        rankings = _load_rankings(ranking_path)

        per_method_rows: list[dict] = []
        pct_top1_values: list[float] = []
        first_rank_values: list[float] = []
        mrr_values: list[float] = []
        r1_values: list[float] = []
        r3_values: list[float] = []
        r5_values: list[float] = []

        for warning_id, warning in by_id.items():
            ranking = rankings.get(warning_id, [])
            essentials = warning.essentials()
            num_essential = len(essentials)
            top1_is_essential = 1 if ranking and ranking[0] in essentials else 0
            first_rank = _first_essential_rank(ranking, essentials)
            rr = 1.0 / first_rank if first_rank > 0 else 0.0

            r1 = recall_at_k(ranking, essentials, 1)
            r3 = recall_at_k(ranking, essentials, 3)
            r5 = recall_at_k(ranking, essentials, 5)
            mrr_value = mrr(ranking, essentials)

            row = {
                "warning_id": warning.warning_id,
                "project": warning.project,
                "method": method,
                "num_essential": num_essential,
                "top1_is_essential": top1_is_essential,
                "first_essential_rank": first_rank,
                "reciprocal_rank": rr,
                "recall@1": r1,
                "recall@3": r3,
                "recall@5": r5,
            }
            warning_rows.append(row)
            per_method_rows.append(row)

            pct_top1_values.append(float(top1_is_essential))
            if first_rank > 0:
                first_rank_values.append(float(first_rank))
            mrr_values.append(mrr_value)
            r1_values.append(r1)
            r3_values.append(r3)
            r5_values.append(r5)

        summary_rows.append(
            {
                "method": method,
                "warnings": len(per_method_rows),
                "pct_top1_essential": _safe_mean(pct_top1_values),
                "mean_first_essential_rank": _safe_mean(first_rank_values),
                "mrr": _safe_mean(mrr_values),
                "recall@1": _safe_mean(r1_values),
                "recall@3": _safe_mean(r3_values),
                "recall@5": _safe_mean(r5_values),
            }
        )

    out_warning = Path(args.out_warning)
    out_warning.parent.mkdir(parents=True, exist_ok=True)
    warning_fields = [
        "warning_id",
        "project",
        "method",
        "num_essential",
        "top1_is_essential",
        "first_essential_rank",
        "reciprocal_rank",
        "recall@1",
        "recall@3",
        "recall@5",
    ]
    with out_warning.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=warning_fields)
        writer.writeheader()
        writer.writerows(warning_rows)

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        "method",
        "warnings",
        "pct_top1_essential",
        "mean_first_essential_rank",
        "mrr",
        "recall@1",
        "recall@3",
        "recall@5",
    ]
    with out_summary.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"wrote {out_warning}")
    print(f"wrote {out_summary}")


if __name__ == "__main__":
    main()
