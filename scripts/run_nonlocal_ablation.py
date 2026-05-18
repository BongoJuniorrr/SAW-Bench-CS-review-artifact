"""Non-local / no-local ablation: test whether methods discriminate when local evidence is removed."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401
from saw_bench_cs.baselines import (
    BM25Ranker,
    EmbeddingRanker,
    LocalFirstRanker,
    TypePriorityRanker,
)
from saw_bench_cs.evaluation.metrics import aggregate
from saw_bench_cs.io import load_warnings, warnings_by_split
from saw_bench_cs.schema import LOCAL_SNIPPET_TYPES, Warning


def run_ablation(
    method_name: str,
    test_warnings: list[Warning],
    prior_warnings: list[Warning],
    condition: str,
) -> dict:
    """Run a single method under one ablation condition."""
    
    # Build ranker
    if method_name == "random":
        ranker = None
    elif method_name == "local_first":
        ranker = LocalFirstRanker()
    elif method_name == "bm25":
        ranker = BM25Ranker()
    elif method_name == "embedding_all-MiniLM-L6-v2":
        ranker = EmbeddingRanker()
    elif method_name == "embedding_hashed_fallback":
        ranker = EmbeddingRanker(allow_hash_fallback=True)
    elif method_name == "type_priority":
        ranker = TypePriorityRanker()
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    # Fit rankers if needed
    fit_warnings = prior_warnings if prior_warnings else test_warnings
    if ranker is not None and hasattr(ranker, 'fit'):
        ranker.fit(fit_warnings)
    
    # Process test warnings based on condition
    modified_warnings = []
    
    for w in test_warnings:
        if condition == "full_candidates":
            modified_warnings.append(w)
        elif condition == "remove_warning_line_and_method":
            # Remove local candidates
            remaining_snippets = [
                s for s in w.candidate_snippets
                if s.type not in LOCAL_SNIPPET_TYPES
            ]
            if not remaining_snippets:
                continue
            
            # Check for essential targets in remaining snippets
            remaining_ids = {s.snippet_id for s in remaining_snippets}
            has_essential = any(
                lbl.snippet_id in remaining_ids and lbl.relevance == "essential"
                for lbl in w.labels
            )
            if not has_essential:
                continue
            
            # Create modified warning
            w_mod = replace(w, candidate_snippets=remaining_snippets)
            modified_warnings.append(w_mod)
        elif condition == "nonlocal_targets_only":
            # Keep full ranking but evaluate only non-local essentials
            modified_warnings.append(w)
        elif condition == "force_one_nonlocal_top3":
            # Will apply post-processing to rankings
            modified_warnings.append(w)
    
    if not modified_warnings:
        return {
            "method": method_name,
            "condition": condition,
            "num_warnings_eval": 0,
            "error": "No warnings remaining after ablation",
        }
    
    # Generate rankings
    rankings = {}
    for w in modified_warnings:
        if method_name == "random":
            ranking = [s.snippet_id for s in w.candidate_snippets]
            random.shuffle(ranking)
        else:
            ranking = ranker.rank(w)
        
        # Apply force_one_nonlocal_top3 if needed
        if condition == "force_one_nonlocal_top3":
            ranking = _force_one_nonlocal_top3(w, ranking)
        
        rankings[w.warning_id] = ranking
    
    # Compute metrics
    if condition == "nonlocal_targets_only":
        metrics = _aggregate_nonlocal_only(modified_warnings, rankings)
    else:
        metrics = aggregate(modified_warnings, rankings)
    
    metrics["num_warnings_eval"] = len(modified_warnings)
    return {
        "method": method_name,
        "condition": condition,
        **metrics,
    }


def _force_one_nonlocal_top3(w: Warning, ranking: list[str]) -> list[str]:
    """Ensure top-3 includes at least one non-local if available."""
    if len(ranking) < 3:
        return ranking
    
    snippet_by_id = {s.snippet_id: s for s in w.candidate_snippets}
    top3 = ranking[:3]
    rest = ranking[3:]
    
    # Find local candidates in top-3
    local_in_top3 = [
        sid for sid in top3
        if snippet_by_id.get(sid, None) and
        snippet_by_id[sid].type in LOCAL_SNIPPET_TYPES
    ]
    
    # Find non-local essentials outside top-3
    nonlocal_essentials = [
        sid for sid in rest
        if snippet_by_id.get(sid, None) and
        snippet_by_id[sid].type not in LOCAL_SNIPPET_TYPES and
        any(lbl.snippet_id == sid and lbl.relevance == "essential"
            for lbl in w.labels)
    ]
    
    if not local_in_top3 or not nonlocal_essentials:
        return ranking
    
    # Replace the last local in top-3 with the highest-ranked non-local essential
    lowest_local = local_in_top3[-1]
    highest_nonlocal = nonlocal_essentials[0]
    
    new_top3 = [s if s != lowest_local else highest_nonlocal for s in top3]
    new_rest = [s if s != highest_nonlocal else lowest_local for s in rest]
    
    return new_top3 + new_rest


def _aggregate_nonlocal_only(warnings: list[Warning], rankings: dict) -> dict:
    """Aggregate metrics but evaluate non-local recall only."""
    from saw_bench_cs.evaluation.metrics import recall_at_k, mrr, ndcg_at_k
    
    results = {
        "recall@1": [],
        "recall@3": [],
        "recall@5": [],
        "mrr": [],
        "ndcg@5": [],
    }
    
    for w in warnings:
        ranking = rankings.get(w.warning_id, [])
        snippet_by_id = {s.snippet_id: s for s in w.candidate_snippets}
        
        # Identify non-local essentials
        nonlocal_essentials = {
            lbl.snippet_id for lbl in w.labels
            if lbl.relevance == "essential" and
            snippet_by_id.get(lbl.snippet_id, None) and
            snippet_by_id[lbl.snippet_id].type not in LOCAL_SNIPPET_TYPES
        }
        
        if not nonlocal_essentials:
            continue
        
        # Create labels map for non-local only
        labels_map = {
            lbl.snippet_id: lbl.relevance for lbl in w.labels
            if lbl.snippet_id in nonlocal_essentials
        }
        
        # Compute metrics
        results["recall@1"].append(recall_at_k(ranking, nonlocal_essentials, 1))
        results["recall@3"].append(recall_at_k(ranking, nonlocal_essentials, 3))
        results["recall@5"].append(recall_at_k(ranking, nonlocal_essentials, 5))
        results["mrr"].append(mrr(ranking, nonlocal_essentials))
        results["ndcg@5"].append(ndcg_at_k(ranking, labels_map, 5))
    
    # Average
    return {
        k: sum(v) / len(v) if v else float("nan")
        for k, v in results.items()
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/saw_bench_cs.jsonl")
    p.add_argument("--prior-split", default="validation")
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="results_reproduced/nonlocal_ablation.csv")
    args = p.parse_args()
    
    random.seed(args.seed)
    
    # Load data
    all_warnings = load_warnings(args.data)
    prior = warnings_by_split(all_warnings, args.prior_split)
    test = warnings_by_split(all_warnings, args.split)
    
    methods = [
        "random",
        "local_first",
        "bm25",
        "embedding_all-MiniLM-L6-v2",
        "type_priority",
    ]
    
    conditions = [
        "full_candidates",
        "remove_warning_line_and_method",
        "nonlocal_targets_only",
        "force_one_nonlocal_top3",
    ]
    
    rows = []
    for condition in conditions:
        for method in methods:
            result = run_ablation(method, test, prior, condition)
            rows.append(result)
            print(f"  {method:30s} × {condition:40s} → {result.get('num_warnings_eval', 0):3d} warnings")
    
    # Write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if rows:
        keys = ["method", "condition", "num_warnings_eval"]
        metric_keys = sorted({k for row in rows for k in row if k not in keys})
        fieldnames = keys + metric_keys
        
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
    
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
