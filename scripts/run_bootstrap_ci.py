"""Bootstrap confidence intervals for ranking metrics on the held-out split."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import _bootstrap  # noqa: F401
from saw_bench_cs.evaluation.metrics import (
    mrr,
    ndcg_at_k,
    non_local_recovery_at_k,
    recall_at_k,
)
from saw_bench_cs.io import load_warnings, warnings_by_split


def _load_rankings(path: Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rankings[row["warning_id"]] = list(row["ranking"])
    return rankings


def _metric_vectors(warnings, rankings):
    vectors = {"recall@3": [], "mrr": [], "ndcg@5": [], "non_local_recovery@3": []}
    for warning in warnings:
        ranking = rankings.get(warning.warning_id, [])
        labels = {lbl.snippet_id: lbl.relevance for lbl in warning.labels}
        essentials = warning.essentials()
        vectors["recall@3"].append(recall_at_k(ranking, essentials, 3))
        vectors["mrr"].append(mrr(ranking, essentials))
        vectors["ndcg@5"].append(ndcg_at_k(ranking, labels, 5))
        vectors["non_local_recovery@3"].append(
            non_local_recovery_at_k(ranking, labels, warning.candidate_snippets, 3)
        )
    return vectors


def _clean_mean(values):
    values = [v for v in values if v == v]
    return sum(values) / len(values) if values else float("nan")


def _ci(samples, alpha=0.05):
    samples = sorted(samples)
    if not samples:
        return float("nan"), float("nan")
    lo = samples[int((alpha / 2) * (len(samples) - 1))]
    hi = samples[int((1 - alpha / 2) * (len(samples) - 1))]
    return lo, hi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/saw_bench_cs.jsonl")
    parser.add_argument("--rankings", default="results_reproduced/baselines/rankings")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="results_reproduced/bootstrap_ci.csv")
    args = parser.parse_args()

    warnings = warnings_by_split(load_warnings(args.data), args.split)
    rng = random.Random(args.seed)
    rankings_dir = Path(args.rankings)
    methods = sorted(p.stem for p in rankings_dir.glob("*.jsonl"))
    method_vectors = {
        method: _metric_vectors(warnings, _load_rankings(rankings_dir / f"{method}.jsonl"))
        for method in methods
    }

    rows: list[dict[str, str]] = []
    for method, vectors in method_vectors.items():
        boot_r3: list[float] = []
        boot_ndcg: list[float] = []
        boot_nonlocal: list[float] = []
        for _ in range(args.n_bootstrap):
            idxs = [rng.randrange(len(warnings)) for _ in warnings]
            boot_r3.append(_clean_mean([vectors["recall@3"][i] for i in idxs]))
            boot_ndcg.append(_clean_mean([vectors["ndcg@5"][i] for i in idxs]))
            boot_nonlocal.append(_clean_mean([vectors["non_local_recovery@3"][i] for i in idxs]))
        rows.append({
            "kind": "method",
            "method": method,
            "other_method": "",
            "metric": "recall@3",
            "mean": f"{_clean_mean(vectors['recall@3']):.4f}",
            "ci_low": f"{_ci(boot_r3)[0]:.4f}",
            "ci_high": f"{_ci(boot_r3)[1]:.4f}",
        })
        rows.append({
            "kind": "method",
            "method": method,
            "other_method": "",
            "metric": "ndcg@5",
            "mean": f"{_clean_mean(vectors['ndcg@5']):.4f}",
            "ci_low": f"{_ci(boot_ndcg)[0]:.4f}",
            "ci_high": f"{_ci(boot_ndcg)[1]:.4f}",
        })
        rows.append({
            "kind": "method",
            "method": method,
            "other_method": "",
            "metric": "non_local_recovery@3",
            "mean": f"{_clean_mean(vectors['non_local_recovery@3']):.4f}",
            "ci_low": f"{_ci(boot_nonlocal)[0]:.4f}",
            "ci_high": f"{_ci(boot_nonlocal)[1]:.4f}",
        })

    pairings = [
        ("type_priority", "local_first"),
        ("type_priority", "bm25"),
        ("type_priority", "embedding_all-MiniLM-L6-v2"),
    ]
    for left, right in pairings:
        if left not in method_vectors or right not in method_vectors:
            continue
        diffs_r3 = []
        diffs_ndcg = []
        diffs_nonlocal = []
        left_vec = method_vectors[left]
        right_vec = method_vectors[right]
        for _ in range(args.n_bootstrap):
            idxs = [rng.randrange(len(warnings)) for _ in warnings]
            diffs_r3.append(_clean_mean([left_vec["recall@3"][i] for i in idxs]) - _clean_mean([right_vec["recall@3"][i] for i in idxs]))
            diffs_ndcg.append(_clean_mean([left_vec["ndcg@5"][i] for i in idxs]) - _clean_mean([right_vec["ndcg@5"][i] for i in idxs]))
            diffs_nonlocal.append(_clean_mean([left_vec["non_local_recovery@3"][i] for i in idxs]) - _clean_mean([right_vec["non_local_recovery@3"][i] for i in idxs]))
        rows.append({
            "kind": "diff",
            "method": left,
            "other_method": right,
            "metric": "recall@3",
            "mean": f"{_clean_mean([a - b for a, b in zip(left_vec['recall@3'], right_vec['recall@3'])]):.4f}",
            "ci_low": f"{_ci(diffs_r3)[0]:.4f}",
            "ci_high": f"{_ci(diffs_r3)[1]:.4f}",
        })
        rows.append({
            "kind": "diff",
            "method": left,
            "other_method": right,
            "metric": "ndcg@5",
            "mean": f"{_clean_mean([a - b for a, b in zip(left_vec['ndcg@5'], right_vec['ndcg@5'])]):.4f}",
            "ci_low": f"{_ci(diffs_ndcg)[0]:.4f}",
            "ci_high": f"{_ci(diffs_ndcg)[1]:.4f}",
        })
        rows.append({
            "kind": "diff",
            "method": left,
            "other_method": right,
            "metric": "non_local_recovery@3",
            "mean": f"{_clean_mean([a - b for a, b in zip(left_vec['non_local_recovery@3'], right_vec['non_local_recovery@3'])]):.4f}",
            "ci_low": f"{_ci(diffs_nonlocal)[0]:.4f}",
            "ci_high": f"{_ci(diffs_nonlocal)[1]:.4f}",
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