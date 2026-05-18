"""Query formulation ablation for the two variants reported in the paper."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401
from saw_bench_cs.baselines import BM25Ranker, EmbeddingRanker
from saw_bench_cs.evaluation.metrics import aggregate
from saw_bench_cs.io import load_warnings, warnings_by_split
from saw_bench_cs.schema import Warning


class ModifiedWarningRanker:
    """Wrapper that ranks after replacing the warning-message query text.

    BM25Ranker and EmbeddingRanker append rule_id and category internally.
    Therefore the default reported query is produced by leaving the warning
    message unchanged. The enriched variant follows the released v23 protocol by
    expanding the warning-message field with the metadata and warning-line text
    used in the shipped query-ablation table.
    """
    
    def __init__(self, base_ranker, query_fn):
        """
        Args:
            base_ranker: BM25Ranker or EmbeddingRanker
            query_fn: Function that takes a Warning and returns query string
        """
        self.base_ranker = base_ranker
        self.query_fn = query_fn
    
    def fit(self, warnings: list[Warning]) -> None:
        """Fit the base ranker."""
        if hasattr(self.base_ranker, 'fit'):
            self.base_ranker.fit(warnings)
    
    def rank(self, warning: Warning) -> list[str]:
        """Rank using modified query."""
        query_text = self.query_fn(warning)
        
        # Create temporary warning with modified message for querying
        temp_warning = replace(warning, warning_message=query_text)
        return self.base_ranker.rank(temp_warning)


def _query_default_message_field(w: Warning) -> str:
    """Message field for the default message + rule + category query."""
    return w.warning_message


def _query_message_rule_category_warning_line(w: Warning) -> str:
    """Message field for message + rule_id + category + warning line."""
    line = w.warning_context.warning_line if w.warning_context else ""
    return f"{w.warning_message} {w.rule_id} {w.category} {line}"


QUERY_VARIANTS = {
    "message_rule_category": _query_default_message_field,
    "message_rule_category_warning_line": _query_message_rule_category_warning_line,
}


def run_query_ablation(
    data_path: str,
    fit_split: str,
    eval_split: str,
    seed: int = 7,
) -> list[dict]:
    """Evaluate BM25 and embedding under different query variants."""
    random.seed(seed)
    
    # Load warnings
    all_warnings = load_warnings(data_path)
    fit = warnings_by_split(all_warnings, fit_split)
    eval_set = warnings_by_split(all_warnings, eval_split)
    
    rows = []
    
    for variant_name, query_fn in QUERY_VARIANTS.items():
        print(f"Testing BM25 with {variant_name}...")
        
        # BM25 with variant
        bm25_ranker = BM25Ranker()
        bm25_ranker.fit(fit)
        bm25_variant = ModifiedWarningRanker(bm25_ranker, query_fn)
        
        # Generate rankings
        rankings = {}
        for w in eval_set:
            rankings[w.warning_id] = bm25_variant.rank(w)
        
        # Compute metrics
        metrics = aggregate(eval_set, rankings)
        row = {
            "method": "bm25",
            "query_variant": variant_name,
            **metrics,
        }
        rows.append(row)
        
        # Embedding with variant
        print(f"Testing embedding with {variant_name}...")
        embedding_ranker = EmbeddingRanker()
        embedding_ranker.fit(fit)
        embedding_variant = ModifiedWarningRanker(embedding_ranker, query_fn)
        
        # Generate rankings
        rankings = {}
        for w in eval_set:
            rankings[w.warning_id] = embedding_variant.rank(w)
        
        # Compute metrics
        metrics = aggregate(eval_set, rankings)
        row = {
            "method": "embedding_all-MiniLM-L6-v2",
            "query_variant": variant_name,
            **metrics,
        }
        rows.append(row)
    
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/saw_bench_cs.jsonl")
    p.add_argument("--fit-split", default="train")
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="results_reproduced/query_ablation.csv")
    args = p.parse_args()
    
    # Run ablation
    rows = run_query_ablation(args.data, args.fit_split, args.split, args.seed)
    
    # Write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if rows:
        fieldnames = ["method", "query_variant"]
        metric_keys = sorted({k for row in rows for k in row if k not in fieldnames})
        fieldnames.extend(metric_keys)
        
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
    
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
