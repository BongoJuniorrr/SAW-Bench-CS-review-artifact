"""Reproduce paper Table 4 (test split context-selection results)."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from saw_bench_cs.evaluation.runner import run_evaluation, write_results
from saw_bench_cs.evaluation.tables import render_table_4
from saw_bench_cs.io import load_warnings


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/saw_bench_cs.jsonl")
    p.add_argument("--fit-split", default="train")
    p.add_argument("--prior-split", default="validation",
                   help="Split used to fit static type-priority priors.")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="results_local/")
    p.add_argument("--include-unavailable-llm-reranker", action="store_true",
                   help="Include llm_reranker even when SAW_LLM_ENDPOINT is unset.")
    args = p.parse_args()

    warnings = load_warnings(args.data)
    results = run_evaluation(
        warnings,
        fit_split=args.fit_split,
        prior_split=args.prior_split,
        eval_split=args.split,
        include_unavailable_llm=args.include_unavailable_llm_reranker,
    )
    out_dir = Path(args.out)
    write_results(results, out_dir)

    print(render_table_4(results))
    print(f"\nwrote {out_dir / 'metrics.csv'} and rankings under {out_dir / 'rankings'}")


if __name__ == "__main__":
    main()
