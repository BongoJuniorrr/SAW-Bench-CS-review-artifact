"""Reproduce paper Table 6 (RQ4 explanation quality study)."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import _bootstrap  # noqa: F401
from saw_bench_cs.baselines import BM25Ranker, EmbeddingRanker, LLMReranker, LocalFirstRanker, TypePriorityRanker
from saw_bench_cs.evaluation.metrics import full_context_tokens, local_only_tokens
from saw_bench_cs.explanation.llm_client import LLMClient
from saw_bench_cs.explanation.prompt_builder import build_prompt
from saw_bench_cs.explanation.rubric_scorer import (
    RubricScores,
    aggregate_scores,
    score_explanation,
)
from saw_bench_cs.io import load_warnings, warnings_by_split


STRATEGIES = ("local_only", "full", "selected")
SELECTOR_CHOICES = ("type_priority", "embedding", "bm25", "local_first", "llm_reranker")


def _boolish(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_scored_rows(path: str | Path) -> dict[str, list]:
    """Load a hand-scored or previously generated RQ4 CSV."""
    scored = {strategy: [] for strategy in STRATEGIES}
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            strategy = row.get("strategy", "")
            if strategy not in scored:
                continue
            scored[strategy].append(RubricScores(
                quality=int(round(float(row["quality"]))),
                mentions_cause=_boolish(row.get("mentions_cause", "")),
                references_evidence=_boolish(row.get("references_evidence", "")),
                no_unsupported_claims=_boolish(row.get("no_unsupported_claims", "")),
                concise=_boolish(row.get("concise", "")),
            ))
    return scored


def print_aggregates(results: dict[str, list]) -> None:
    print("Aggregated rubric scores per strategy:")
    for strategy in STRATEGIES:
        agg = aggregate_scores(results[strategy])
        print(f"  {strategy:>11s}  quality={agg['quality']:.2f}  "
              f"unsupported={agg['unsupported_claim_rate']:.2f}")


def make_selector_ranker(selector: str, train: list, priors: list):
    if selector == "type_priority":
        ranker = TypePriorityRanker()
        ranker.fit(priors)
        return ranker
    if selector == "embedding":
        return EmbeddingRanker()
    if selector == "bm25":
        ranker = BM25Ranker()
        ranker.fit(train)
        return ranker
    if selector == "local_first":
        return LocalFirstRanker()
    if selector == "llm_reranker":
        ranker = LLMReranker()
        ranker.fit(train)
        return ranker
    raise ValueError(f"unknown selector {selector!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/saw_bench_cs.jsonl")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--selector",
        default="type_priority",
        choices=list(SELECTOR_CHOICES),
        help="Ranker used to select the top-3 snippets for the selected-context prompt.",
    )
    p.add_argument("--scores", default=None,
                   help="Existing hand-scored CSV to aggregate instead of calling an LLM.")
    p.add_argument("--out", default="results_local/explanations.csv")
    args = p.parse_args()

    if args.scores:
        print_aggregates(load_scored_rows(args.scores))
        return

    rng = random.Random(args.seed)
    all_warnings = load_warnings(args.data)
    warnings = warnings_by_split(all_warnings, args.split)
    train = warnings_by_split(all_warnings, "train")
    priors = warnings_by_split(all_warnings, "validation")
    rng.shuffle(warnings)
    sample = warnings[: args.n]

    selector_ranker = make_selector_ranker(args.selector, train, priors)
    client = LLMClient()
    if not client.is_configured():
        raise SystemExit(
            "RQ4 generation requires SAW_LLM_ENDPOINT. Pass --scores to aggregate "
            "a hand-scored CSV without calling an LLM."
        )

    results = {strategy: [] for strategy in STRATEGIES}
    rows: list[dict] = []
    for w in sample:
        selected = selector_ranker.rank(w)[:3]
        for strategy in STRATEGIES:
            prompt, snippets = build_prompt(
                w, strategy, selected_ids=selected, top_k=3,
            )
            explanation = client.complete(prompt) or ""
            scores = score_explanation(
                explanation, snippets, w.warning_message,
                rule_id=w.rule_id, client=client,
            )
            results[strategy].append(scores)
            rows.append({
                "warning_id": w.warning_id,
                "strategy": strategy,
                "quality": scores.quality,
                "mentions_cause": scores.mentions_cause,
                "references_evidence": scores.references_evidence,
                "no_unsupported_claims": scores.no_unsupported_claims,
                "concise": scores.concise,
                "tokens": (
                    full_context_tokens(w) if strategy == "full" else
                    local_only_tokens(w) if strategy == "local_only" else
                    sum(s.token_count for s in snippets)
                ),
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("No explanation rows were produced; check --data, --split, and --n.")
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print_aggregates(results)


if __name__ == "__main__":
    main()
