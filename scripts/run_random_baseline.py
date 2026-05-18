"""Compute a random-permutation baseline for context selection metrics."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import _bootstrap  # noqa: F401
from saw_bench_cs.evaluation.metrics import aggregate
from saw_bench_cs.io import load_warnings, warnings_by_split


DEFAULT_METRICS = (
    "recall@1",
    "recall@3",
    "recall@5",
    "mrr",
    "ndcg@5",
    "non_local_recovery@3",
    "avg_tokens@3",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/saw_bench_cs.jsonl")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-shuffles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="results_local/random_baseline.json")
    args = parser.parse_args()

    warnings = warnings_by_split(load_warnings(args.data), args.split)
    values = {metric: [] for metric in DEFAULT_METRICS}

    for i in range(args.n_shuffles):
        rng = random.Random(args.seed + i)
        rankings = {}
        for warning in warnings:
            ids = [snippet.snippet_id for snippet in warning.candidate_snippets]
            rng.shuffle(ids)
            rankings[warning.warning_id] = ids
        metrics = aggregate(warnings, rankings)
        for metric in DEFAULT_METRICS:
            values[metric].append(metrics[metric])

    summary = {
        "split": args.split,
        "seed": args.seed,
        "n_shuffles": args.n_shuffles,
        "metrics": {
            metric: {
                "mean": round(statistics.mean(vals), 4),
                "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            }
            for metric, vals in values.items()
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
