"""Run baselines and aggregate metrics into Table 4-style results."""

from __future__ import annotations

import csv
import json
import os
import warnings as py_warnings
from dataclasses import dataclass
from pathlib import Path

from ..baselines import (
    BM25Ranker,
    EmbeddingRanker,
    LLMReranker,
    LocalFirstRanker,
    Ranker,
    TypePriorityRanker,
)
from ..io import warnings_by_split
from ..schema import Warning
from .metrics import aggregate


@dataclass
class EvaluationResult:
    method: str
    metrics: dict[str, float]
    rankings: dict[str, list[str]]


def default_rankers(
    fit_warnings: list[Warning],
    *,
    prior_warnings: list[Warning] | None = None,
    include_unavailable_llm: bool = False,
) -> list[Ranker]:
    """Standard set of five baselines (paper §5.2)."""
    type_priority = TypePriorityRanker()
    type_priority.fit(prior_warnings if prior_warnings is not None else fit_warnings)

    # Allow overriding embedding model/device via environment for GPU runs.
    emb_device = os.environ.get("SAW_EMBEDDING_DEVICE", "cpu")
    emb_model = os.environ.get(
        "SAW_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    rankers: list[Ranker] = [
        LocalFirstRanker(),
        BM25Ranker(),
        EmbeddingRanker(model_name=emb_model, device=emb_device),
        type_priority,
    ]
    if os.environ.get("SAW_LLM_ENDPOINT") or include_unavailable_llm:
        llm_model = os.environ.get("SAW_LLM_MODEL")
        if llm_model:
            rankers.append(LLMReranker(model=llm_model))
        else:
            rankers.append(LLMReranker())
    else:
        py_warnings.warn(
            "Skipping llm_reranker because SAW_LLM_ENDPOINT is not configured.",
            RuntimeWarning,
        )
    return rankers


def run_evaluation(
    warnings: list[Warning],
    *,
    fit_split: str = "train",
    prior_split: str = "validation",
    eval_split: str = "test",
    rankers: list[Ranker] | None = None,
    include_unavailable_llm: bool = False,
) -> list[EvaluationResult]:
    """Fit rankers on one split, evaluate on another, return metric tables."""
    train = warnings_by_split(warnings, fit_split)
    priors = warnings_by_split(warnings, prior_split)
    eval_set = warnings_by_split(warnings, eval_split)
    if rankers is None:
        rankers = default_rankers(
            train,
            prior_warnings=priors,
            include_unavailable_llm=include_unavailable_llm,
        )
    else:
        for r in rankers:
            r.fit(train)

    results: list[EvaluationResult] = []
    for ranker in rankers:
        rankings: dict[str, list[str]] = {}
        for w in eval_set:
            rankings[w.warning_id] = ranker.rank(w)
        metrics = aggregate(eval_set, rankings)
        results.append(EvaluationResult(ranker.name, metrics, rankings))
    return results


def write_results(results: list[EvaluationResult], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Aggregated metric table (one row per method).
    metric_keys: list[str] = []
    for r in results:
        for k in r.metrics:
            if k not in metric_keys:
                metric_keys.append(k)
    table_path = out / "metrics.csv"
    with table_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method", *metric_keys])
        for r in results:
            writer.writerow([r.method, *(r.metrics.get(k, "") for k in metric_keys)])

    # Per-warning rankings — one JSONL per method, useful for diagnostics replay.
    rankings_dir = out / "rankings"
    rankings_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        path = rankings_dir / f"{r.method}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for warning_id, ranking in r.rankings.items():
                fh.write(json.dumps({"warning_id": warning_id, "ranking": ranking}))
                fh.write("\n")
