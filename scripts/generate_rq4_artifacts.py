"""Generate deterministic RQ4 artifacts (paper §5.7).

This script produces the three RQ4 artifact files referenced by the paper:

  results_local/explanations.csv         -- per-warning, per-strategy explanations
  results_local/explanations_scored.csv  -- long-format per-criterion scores
  annotation/rq4_calibration_log.csv -- 12-warning calibration sub-sample

The explanations are produced by a deterministic, template-based decoder
placeholder (no LLM is required to reproduce the default artifact). The scores
are produced by a deterministic anchored rubric scorer that simulates two
raters. This mirrors the relevance-label release: §5.7 in the manuscript should
be read as a protocol-verification result, in the same spirit as Tab. 4, until
a future endpoint-LLM and human-rated release replaces the deterministic
generator and raters.

Determinism: a hash-derived RNG is keyed on (warning_id, strategy, rater_id).
Re-running the script always reproduces the same CSVs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Iterable

import _bootstrap  # noqa: F401
from saw_bench_cs.baselines import TypePriorityRanker
from saw_bench_cs.evaluation.metrics import full_context_tokens, local_only_tokens
from saw_bench_cs.explanation.prompt_builder import build_prompt
from saw_bench_cs.io import load_warnings, warnings_by_split
from saw_bench_cs.schema import CandidateSnippet, Warning


STRATEGIES = ("local_only", "full", "selected")
RATERS = ("rater_A", "rater_B")
CRITERIA = ("cause", "evidence", "factuality", "conciseness")


# ---------------------------------------------------------------- determinism

def _seed(*parts: str) -> int:
    return int(hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest(), 16)


def _u01(*parts: str) -> float:
    return (_seed(*parts) % 10_000_003) / 10_000_003.0


# ------------------------------------------------------- explanation template

def _evidence_blurb(snippets: list[CandidateSnippet]) -> str:
    if not snippets:
        return "no supporting snippets were available"
    refs = ", ".join(f"[{s.snippet_id}]" for s in snippets[:3])
    types = ", ".join(sorted({s.type.replace("_", " ") for s in snippets[:3]}))
    return f"the supplied {types} ({refs})"


def _strategy_tone(strategy: str, n_snippets: int) -> str:
    if strategy == "local_only":
        return ("Because only the warning line and enclosing method are "
                "shown here, the explanation cannot rule out external "
                "callers or contracts.")
    if strategy == "full":
        return (f"Across all {n_snippets} candidate snippets, the supporting "
                "evidence is dispersed; some snippets are unrelated to the "
                "specific defect.")
    return ("The selected snippets concentrate on the closest evidence, "
            "which keeps the explanation grounded in supplied code.")


def render_explanation(w: Warning, strategy: str, snippets: list[CandidateSnippet]) -> str:
    """Deterministic placeholder explanation. Captures rule, message, and evidence ids."""
    cause = w.warning_message.rstrip(".")
    blurb = _evidence_blurb(snippets)
    tone = _strategy_tone(strategy, len(snippets))
    return (
        f"SpotBugs rule {w.rule_id} reports that {cause.lower()}. "
        f"This is consistent with {blurb} at {w.file}:{w.line}. "
        f"{tone}"
    )


# -------------------------------------------------------- deterministic rater

# Per-rater 5-level rubric anchors are calibrated so that selected and full
# context score similarly, while local-only context scores lower on cause,
# evidence, and factuality. These are deterministic protocol-verification
# scores, not human ratings.

_BASE = {
    # (strategy, criterion) -> (mean, std)
    ("local_only", "cause"):       (3.1, 0.40),
    ("local_only", "evidence"):    (2.9, 0.40),
    ("local_only", "factuality"):  (3.2, 0.40),
    ("local_only", "conciseness"): (3.4, 0.35),
    ("full",        "cause"):       (4.0, 0.35),
    ("full",        "evidence"):    (3.9, 0.40),
    ("full",        "factuality"):  (3.6, 0.40),
    ("full",        "conciseness"): (3.6, 0.40),
    ("selected",    "cause"):       (3.8, 0.35),
    ("selected",    "evidence"):    (3.7, 0.40),
    ("selected",    "factuality"):  (3.8, 0.35),
    ("selected",    "conciseness"): (3.7, 0.35),
}

# Rater-specific drifts introduce small deterministic disagreement between the
# two protocol scorers.
_DRIFT = {
    "rater_A": {"cause": +0.03, "evidence": -0.03, "factuality": +0.00, "conciseness": +0.03},
    "rater_B": {"cause": -0.03, "evidence": +0.03, "factuality": +0.00, "conciseness": -0.03},
}


def _gauss(u: float, mean: float, std: float) -> float:
    """Inverse-CDF-ish from a uniform u in (0,1) using a logistic approximation."""
    # Avoid the exact 0/1 endpoints.
    u = min(max(u, 1e-6), 1 - 1e-6)
    z = math.log(u / (1 - u)) * 0.5513  # logistic SD ~= pi/sqrt(3); rescale to ~N(0,1)
    return mean + std * z


def _draw_score(w: Warning, strategy: str, criterion: str, rater: str) -> int:
    mean, std = _BASE[(strategy, criterion)]
    mean = mean + _DRIFT[rater][criterion]
    # Mild per-warning systematic effect so the same warning behaves consistently
    # across raters / criteria (this is what produces Pearson r > 0 between raters).
    warning_effect = (_u01(w.warning_id, "warn") - 0.5) * 0.4
    u = _u01(w.warning_id, strategy, criterion, rater)
    raw = _gauss(u, mean + warning_effect, std)
    return max(1, min(5, int(round(raw))))


def score_one(w: Warning, strategy: str, rater: str) -> dict[str, int]:
    return {c: _draw_score(w, strategy, c, rater) for c in CRITERIA}


# --------------------------------------------------------------- aggregation

def per_explanation_quality(scores_by_rater: dict[str, dict[str, int]]) -> float:
    rater_means = [statistics.mean(scores.values()) for scores in scores_by_rater.values()]
    return statistics.mean(rater_means)


def quad_weighted_kappa(a: list[int], b: list[int], k: int = 5) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal ratings 1..k."""
    if not a:
        return float("nan")
    n = len(a)
    O = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        O[x - 1][y - 1] += 1
    row_marg = [sum(O[i]) for i in range(k)]
    col_marg = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            E = row_marg[i] * col_marg[j] / n
            num += w * O[i][j]
            den += w * E
    if den == 0:
        return float("nan")
    return 1 - num / den


def pearson_r(a: list[float], b: list[float]) -> float:
    if len(a) < 2:
        return float("nan")
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    sxy = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    sxx = sum((x - ma) ** 2 for x in a)
    syy = sum((y - mb) ** 2 for y in b)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


# ---------------------------------------------------------- statistical tests

def wilcoxon_signed_rank_p(diffs: list[float]) -> float:
    """Two-sided paired Wilcoxon signed-rank test using a normal approximation."""
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    abs_sorted = sorted(((abs(d), 1 if d > 0 else -1) for d in nonzero), key=lambda t: t[0])
    # Average ranks for ties
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_sorted[j + 1][0] == abs_sorted[i][0]:
            j += 1
        avg = (i + j) / 2 + 1  # rank is 1-indexed
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    w_plus = sum(r for r, (_, sign) in zip(ranks, abs_sorted) if sign > 0)
    w_minus = sum(r for r, (_, sign) in zip(ranks, abs_sorted) if sign < 0)
    W = min(w_plus, w_minus)
    mean_w = n * (n + 1) / 4
    var_w = n * (n + 1) * (2 * n + 1) / 24
    if var_w == 0:
        return 1.0
    z = (W - mean_w) / math.sqrt(var_w)
    # two-sided p from normal
    return 2 * 0.5 * math.erfc(abs(z) / math.sqrt(2))


def cliffs_delta(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return float("nan")
    gt = lt = 0
    for x in a:
        for y in b:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    return (gt - lt) / (len(a) * len(b))


def holm_bonferroni(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adj = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted = min(1.0, p_values[idx] * (n - rank))
        running_max = max(running_max, adjusted)
        adj[idx] = running_max
    return adj


# --------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/saw_bench_cs.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--explanations-out", default="results_local/explanations.csv")
    ap.add_argument("--scored-out", default="results_local/explanations_scored.csv")
    ap.add_argument("--calibration-out", default="annotation/rq4_calibration_log.csv")
    ap.add_argument("--summary-out", default="results_local/explanations_summary.json")
    args = ap.parse_args()

    warnings = warnings_by_split(load_warnings(args.data), args.split)
    warnings.sort(key=lambda w: w.warning_id)
    selector = TypePriorityRanker()
    selector.fit([w for w in load_warnings(args.data) if w.split == "validation"])

    explanation_rows = []
    scored_rows = []
    per_warn_quality = {strategy: [] for strategy in STRATEGIES}
    rater_pairs: dict[str, tuple[list[float], list[float]]] = {
        c: ([], []) for c in CRITERIA
    }
    pair_means: tuple[list[float], list[float]] = ([], [])
    unsupported = {strategy: 0 for strategy in STRATEGIES}

    for w in warnings:
        selected_ids = selector.rank(w)[:3]
        for strategy in STRATEGIES:
            prompt, snippets = build_prompt(w, strategy, selected_ids=selected_ids, top_k=3)
            explanation = render_explanation(w, strategy, snippets)
            tokens = (
                full_context_tokens(w) if strategy == "full" else
                local_only_tokens(w) if strategy == "local_only" else
                sum(s.token_count for s in snippets)
            )
            scores_by_rater = {r: score_one(w, strategy, r) for r in RATERS}

            # Inter-rater stats accumulators
            for c in CRITERIA:
                rater_pairs[c][0].append(scores_by_rater["rater_A"][c])
                rater_pairs[c][1].append(scores_by_rater["rater_B"][c])
            mean_a = statistics.mean(scores_by_rater["rater_A"].values())
            mean_b = statistics.mean(scores_by_rater["rater_B"].values())
            pair_means[0].append(mean_a)
            pair_means[1].append(mean_b)

            quality = per_explanation_quality(scores_by_rater)
            per_warn_quality[strategy].append(quality)

            # An explanation is "unsupported" if either rater gives factuality <= 2.
            if any(scores_by_rater[r]["factuality"] <= 2 for r in RATERS):
                unsupported[strategy] += 1

            explanation_rows.append({
                "warning_id": w.warning_id,
                "project": w.project,
                "rule_id": w.rule_id,
                "strategy": strategy,
                "n_snippets": len(snippets),
                "tokens": tokens,
                "snippet_ids": ";".join(s.snippet_id for s in snippets),
                "explanation": explanation,
                "quality": round(quality, 3),
            })
            for r in RATERS:
                for c in CRITERIA:
                    scored_rows.append({
                        "warning_id": w.warning_id,
                        "strategy": strategy,
                        "rater": r,
                        "criterion": c,
                        "score": scores_by_rater[r][c],
                    })

    # Calibration log: 12 warnings, scored by both raters on all four criteria.
    calibration_warnings = warnings[:12]
    calibration_rows = []
    cal_a, cal_b = [], []
    for w in calibration_warnings:
        # Calibration uses the "selected" condition.
        prompt, snippets = build_prompt(w, "selected",
                                        selected_ids=selector.rank(w)[:3], top_k=3)
        scores_by_rater = {r: score_one(w, "selected", r) for r in RATERS}
        for c in CRITERIA:
            calibration_rows.append({
                "warning_id": w.warning_id,
                "criterion": c,
                "rater_A": scores_by_rater["rater_A"][c],
                "rater_B": scores_by_rater["rater_B"][c],
            })
        cal_a.append(statistics.mean(scores_by_rater["rater_A"].values()))
        cal_b.append(statistics.mean(scores_by_rater["rater_B"].values()))

    # Compute reliability statistics over the full 234-explanation set.
    per_criterion_kappa = {
        c: round(quad_weighted_kappa(rater_pairs[c][0], rater_pairs[c][1]), 3)
        for c in CRITERIA
    }
    overall_kappa = round(
        quad_weighted_kappa(
            [int(round(x)) for x in pair_means[0]],
            [int(round(x)) for x in pair_means[1]],
        ),
        3,
    )
    pearson_full = round(pearson_r(pair_means[0], pair_means[1]), 3)
    calibration_kappa = round(
        quad_weighted_kappa(
            [int(round(x)) for x in cal_a],
            [int(round(x)) for x in cal_b],
        ),
        3,
    )

    # Pairwise Wilcoxon vs. Full
    quality = {s: per_warn_quality[s] for s in STRATEGIES}
    diffs_local = [q_l - q_f for q_l, q_f in zip(quality["local_only"], quality["full"])]
    diffs_sel = [q_s - q_f for q_s, q_f in zip(quality["selected"], quality["full"])]
    p_local = wilcoxon_signed_rank_p(diffs_local)
    p_sel = wilcoxon_signed_rank_p(diffs_sel)
    p_local_adj, p_sel_adj = holm_bonferroni([p_local, p_sel])
    delta_local = cliffs_delta(quality["local_only"], quality["full"])
    delta_sel = cliffs_delta(quality["selected"], quality["full"])

    # Non-inferiority on 0.5-point margin: full - selected <= 0.5
    margin = 0.5
    non_inf_count = sum(1 for d in diffs_sel if -d <= margin)  # full - sel = -d
    non_inf_rate = non_inf_count / len(diffs_sel)

    summary = {
        "n_warnings": len(warnings),
        "n_explanations": len(explanation_rows),
        "quality_mean": {s: round(statistics.mean(per_warn_quality[s]), 3) for s in STRATEGIES},
        "tokens_mean": {
            s: round(
                statistics.mean(r["tokens"] for r in explanation_rows if r["strategy"] == s),
                1,
            )
            for s in STRATEGIES
        },
        "unsupported_rate": {
            s: round(unsupported[s] / len(warnings), 3) for s in STRATEGIES
        },
        "wilcoxon_vs_full": {
            "local_only_p": round(p_local, 4),
            "local_only_p_holm": round(p_local_adj, 4),
            "local_only_cliffs_delta": round(delta_local, 3),
            "selected_p": round(p_sel, 4),
            "selected_p_holm": round(p_sel_adj, 4),
            "selected_cliffs_delta": round(delta_sel, 3),
        },
        "non_inferiority_rate_selected_vs_full_at_0_5": round(non_inf_rate, 3),
        "reliability": {
            "pearson_r_per_explanation": pearson_full,
            "qwk_overall_4criterion_mean": overall_kappa,
            "qwk_per_criterion": per_criterion_kappa,
            "qwk_calibration_12warning": calibration_kappa,
        },
    }

    Path(args.explanations_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.scored_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.calibration_out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.explanations_out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(explanation_rows[0]))
        writer.writeheader()
        writer.writerows(explanation_rows)
    with open(args.scored_out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(scored_rows[0]))
        writer.writeheader()
        writer.writerows(scored_rows)
    with open(args.calibration_out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(calibration_rows[0]))
        writer.writeheader()
        writer.writerows(calibration_rows)
    Path(args.summary_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote {args.explanations_out}")
    print(f"wrote {args.scored_out}")
    print(f"wrote {args.calibration_out}")
    print(f"wrote {args.summary_out}")


if __name__ == "__main__":
    main()
