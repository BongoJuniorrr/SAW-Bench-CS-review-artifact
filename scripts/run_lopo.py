"""Leave-one-project-out (LOPO) evaluation across all analyzed projects.

For each project p in the dataset, fit the baselines on the remaining projects
and evaluate on p alone. Reports per-project R@3, MRR, and aggregate dispersion.
This script supports the threats-to-validity discussion of test-set size by
showing that the headline R@3 estimate is stable to which projects are held out.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
from saw_bench_cs.baselines import (
    BM25Ranker,
    EmbeddingRanker,
    LocalFirstRanker,
    Ranker,
    TypePriorityRanker,
)
from saw_bench_cs.evaluation.metrics import aggregate
from saw_bench_cs.io import load_warnings
from saw_bench_cs.schema import Warning


def make_rankers(fit: list[Warning]) -> list[Ranker]:
    type_priority = TypePriorityRanker()
    type_priority.fit(fit)
    return [LocalFirstRanker(), BM25Ranker(), EmbeddingRanker(), type_priority]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/saw_bench_cs.jsonl")
    p.add_argument("--out", default="results_local/lopo.csv")
    args = p.parse_args()

    warnings = load_warnings(args.data)
    by_project: dict[str, list[Warning]] = defaultdict(list)
    for w in warnings:
        by_project[w.project].append(w)
    projects = sorted(by_project)

    rows = [("project", "method", "n", "recall@3", "mrr", "ndcg@5", "non_local_recovery@3")]
    method_to_r3: dict[str, list[float]] = defaultdict(list)
    for held_out in projects:
        fit = [w for w in warnings if w.project != held_out]
        eval_set = by_project[held_out]
        for ranker in make_rankers(fit):
            ranker.fit(fit)
            rankings = {w.warning_id: ranker.rank(w) for w in eval_set}
            metrics = aggregate(eval_set, rankings)
            rows.append(
                (
                    held_out,
                    ranker.name,
                    str(len(eval_set)),
                    f"{metrics.get('recall@3', 0.0):.3f}",
                    f"{metrics.get('mrr', 0.0):.3f}",
                    f"{metrics.get('ndcg@5', 0.0):.3f}",
                    f"{metrics.get('non_local_recovery@3', metrics.get('nonlocal_recall@3', 0.0)):.3f}",
                )
            )
            method_to_r3[ranker.name].append(metrics.get("recall@3", 0.0))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)

    print(f"wrote {out}")
    print("\nLOPO summary (R@3 across projects):")
    print(f"{'method':<16} {'n_proj':>6} {'mean':>6} {'min':>6} {'max':>6} {'sd':>6}")
    for method, vals in sorted(method_to_r3.items()):
        if not vals:
            continue
        print(
            f"{method:<16} {len(vals):>6d} {statistics.mean(vals):>6.3f} "
            f"{min(vals):>6.3f} {max(vals):>6.3f} "
            f"{(statistics.stdev(vals) if len(vals) > 1 else 0.0):>6.3f}"
        )


if __name__ == "__main__":
    main()
