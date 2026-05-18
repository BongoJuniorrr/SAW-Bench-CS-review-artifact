"""Compute Table 3 diagnostic metrics from saved per-method ranking files.

Each input ranking file is JSONL of {warning_id, ranking}. The script joins
those rankings against the dataset, computes the diagnostic metrics, and
writes a CSV summary.
"""

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
    p.add_argument("--rankings", default="results_local/rankings/")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="results_local/diagnostics.csv")
    args = p.parse_args()

    warnings = warnings_by_split(load_warnings(args.data), args.split)
    rankings_dir = Path(args.rankings)
    methods = sorted(p for p in rankings_dir.glob("*.jsonl"))

    rows: list[dict] = []
    metric_keys: list[str] = []
    for path in methods:
        rankings = _load_rankings(path)
        metrics = aggregate(warnings, rankings)
        row = {"method": path.stem, **metrics}
        rows.append(row)
        for k in metrics:
            if k not in metric_keys:
                metric_keys.append(k)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method", *metric_keys])
        for row in rows:
            writer.writerow([row["method"], *(row.get(k, "") for k in metric_keys)])

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
