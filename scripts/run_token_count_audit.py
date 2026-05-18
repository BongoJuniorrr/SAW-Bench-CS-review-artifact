"""Audit token-count distributions for full, local-only, and selected top-3 contexts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from saw_bench_cs.io import load_warnings, warnings_by_split
from saw_bench_cs.schema import LOCAL_SNIPPET_TYPES


def _load_rankings(path: Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rankings[row["warning_id"]] = list(row["ranking"])
    return rankings


def _percentile(sorted_values: list[int], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    if pct <= 0:
        return float(sorted_values[0])
    if pct >= 100:
        return float(sorted_values[-1])
    index = int(round((pct / 100.0) * (len(sorted_values) - 1)))
    return float(sorted_values[index])


def _summarize(values: list[int]) -> dict[str, float]:
    if not values:
        return {
            "warnings": 0,
            "mean_tokens": float("nan"),
            "median_tokens": float("nan"),
            "p95_tokens": float("nan"),
            "min_tokens": float("nan"),
            "max_tokens": float("nan"),
        }
    sorted_values = sorted(values)
    n = len(values)
    mean_tokens = float(sum(values)) / float(n)
    median_tokens = _percentile(sorted_values, 50)
    p95_tokens = _percentile(sorted_values, 95)
    return {
        "warnings": n,
        "mean_tokens": mean_tokens,
        "median_tokens": median_tokens,
        "p95_tokens": p95_tokens,
        "min_tokens": float(sorted_values[0]),
        "max_tokens": float(sorted_values[-1]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/saw_bench_cs.jsonl")
    p.add_argument("--rankings", default="results_reproduced/baselines/rankings")
    p.add_argument("--out", default="results_reproduced/token_count_audit.csv")
    args = p.parse_args()

    all_warnings = load_warnings(args.data)
    splits = ["train", "validation", "test"]

    ranking_paths = sorted(Path(args.rankings).glob("*.jsonl"))
    ranking_maps: dict[str, dict[str, list[str]]] = {
        pth.stem: _load_rankings(pth) for pth in ranking_paths
    }

    rows: list[dict] = []

    for split in splits:
        warnings = warnings_by_split(all_warnings, split)

        full_values: list[int] = []
        local_values: list[int] = []
        for warning in warnings:
            full_values.append(sum(s.token_count for s in warning.candidate_snippets))
            local_values.append(
                sum(
                    s.token_count
                    for s in warning.candidate_snippets
                    if s.type in LOCAL_SNIPPET_TYPES
                )
            )

        for condition_name, values in [
            ("full_context", full_values),
            ("local_only", local_values),
        ]:
            summary = _summarize(values)
            rows.append(
                {
                    "split": split,
                    "condition_or_method": condition_name,
                    **summary,
                }
            )

        warning_ids = {w.warning_id for w in warnings}
        for method, rankings in sorted(ranking_maps.items()):
            covered_ids = warning_ids & set(rankings.keys())
            if not covered_ids:
                continue
            selected_values: list[int] = []
            for warning in warnings:
                if warning.warning_id not in covered_ids:
                    continue
                by_id = {s.snippet_id: s for s in warning.candidate_snippets}
                ranking = rankings.get(warning.warning_id, [])
                selected_values.append(
                    sum(by_id[sid].token_count for sid in ranking[:3] if sid in by_id)
                )
            summary = _summarize(selected_values)
            rows.append(
                {
                    "split": split,
                    "condition_or_method": method,
                    **summary,
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "condition_or_method",
        "warnings",
        "mean_tokens",
        "median_tokens",
        "p95_tokens",
        "min_tokens",
        "max_tokens",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
