"""Evaluation metrics and runner for SAW-Bench-CS (paper §5)."""

from .metrics import (
    average_tokens,
    coverage_at_k,
    distractor_rate_at_k,
    mrr,
    ndcg_at_k,
    non_local_recovery_at_k,
    recall_at_k,
    token_normalized_utility,
)
from .runner import EvaluationResult, run_evaluation

__all__ = [
    "average_tokens",
    "coverage_at_k",
    "distractor_rate_at_k",
    "mrr",
    "ndcg_at_k",
    "non_local_recovery_at_k",
    "recall_at_k",
    "token_normalized_utility",
    "EvaluationResult",
    "run_evaluation",
]
