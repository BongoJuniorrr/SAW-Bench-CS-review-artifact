"""Run baselines on hard-negative stress test dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
from saw_bench_cs.baselines import (
    BM25Ranker,
    EmbeddingRanker,
    LocalFirstRanker,
    TypePriorityRanker,
)
from saw_bench_cs.evaluation.metrics import (
    distractor_rate_at_k,
    recall_at_k,
    ndcg_at_k,
    mrr,
    aggregate,
)
from saw_bench_cs.io import load_warnings, warnings_by_split


def run_hard_negative_eval(
    data_path: str,
    fit_split: str,
    eval_split: str,
    seed: int = 7,
) -> list[dict]:
    """Evaluate all methods on hard-negative dataset."""
    import random
    random.seed(seed)
    
    # Load warnings
    all_warnings = load_warnings(data_path)
    fit = warnings_by_split(all_warnings, fit_split)
    eval_set = warnings_by_split(all_warnings, eval_split)
    
    methods = [
        ("random", None),
        ("local_first", LocalFirstRanker()),
        ("bm25", BM25Ranker()),
        ("embedding_all-MiniLM-L6-v2", EmbeddingRanker()),
        ("type_priority", TypePriorityRanker()),
    ]
    
    rows = []
    
    for method_name, ranker in methods:
        print(f"Evaluating {method_name}...")
        
        # Fit ranker if needed
        if ranker is not None and hasattr(ranker, 'fit'):
            ranker.fit(fit)
        
        # Generate rankings
        rankings = {}
        for w in eval_set:
            if method_name == "random":
                ranking = [s.snippet_id for s in w.candidate_snippets]
                import random
                random.shuffle(ranking)
            else:
                ranking = ranker.rank(w)
            rankings[w.warning_id] = ranking
        
        # Compute metrics
        metrics = aggregate(eval_set, rankings)
        
        # Additional hard-negative metrics
        hard_neg_rate_3 = []
        hard_neg_rate_5 = []
        
        for w in eval_set:
            ranking = rankings.get(w.warning_id, [])
            labels = {lbl.snippet_id: lbl.relevance for lbl in w.labels}
            
            # Count hard negatives (hnXX snippets)
            for k in [3, 5]:
                top_k = ranking[:k]
                hard_negs = sum(1 for sid in top_k if sid.startswith("hn"))
                hard_neg_rate_3.append(hard_negs / k) if k == 3 else hard_neg_rate_5.append(hard_negs / k)
        
        # Average hard-neg metrics
        avg_hard_neg_3 = sum(hard_neg_rate_3) / len(hard_neg_rate_3) if hard_neg_rate_3 else 0.0
        avg_hard_neg_5 = sum(hard_neg_rate_5) / len(hard_neg_rate_5) if hard_neg_rate_5 else 0.0
        
        row = {
            "method": method_name,
            "hard_negative_rate@3": avg_hard_neg_3,
            "hard_negative_rate@5": avg_hard_neg_5,
            **metrics,
        }
        rows.append(row)
    
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/saw_bench_cs_hardneg.jsonl")
    p.add_argument("--fit-split", default="train")
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="results_reproduced/hardneg_metrics.csv")
    args = p.parse_args()
    
    # Run evaluation
    rows = run_hard_negative_eval(args.data, args.fit_split, args.split, args.seed)
    
    # Write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if rows:
        fieldnames = list(rows[0].keys())
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
