"""Run additional experiments E2, E3, E4, E5, E6, E7, E8 for reviewer responses.

Output: results_extra/ directory with one CSV per experiment.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401

from saw_bench_cs.io import load_warnings, warnings_by_split
from saw_bench_cs.baselines.type_priority import TypePriorityRanker, fit_priors
from saw_bench_cs.evaluation.metrics import (
    recall_at_k, mrr, ndcg_at_k, non_local_recovery_at_k, aggregate
)
from saw_bench_cs.schema import SNIPPET_TYPES, LOCAL_SNIPPET_TYPES, RELEVANCE_GAIN

DATA_PATH = "data/saw_bench_cs.jsonl"
HARDNEG_PATH = "data/saw_bench_cs_hardneg.jsonl"
RANKINGS_DIR = Path("results_reproduced/rankings")
OUT_DIR = Path("results_extra")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42
rng = random.Random(RNG_SEED)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN.findall(text or "")]


def _safe_mean(values):
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else float("nan")


def _eval_metrics(warnings, rankings_dict):
    """Return dict of aggregate metrics for a set of warnings and rankings."""
    r3, ndcg, nlr = [], [], []
    for w in warnings:
        ranking = rankings_dict.get(w.warning_id, [])
        labels = {lbl.snippet_id: lbl.relevance for lbl in w.labels}
        essentials = w.essentials()
        r3.append(recall_at_k(ranking, essentials, 3))
        ndcg.append(ndcg_at_k(ranking, labels, 5))
        nlr.append(non_local_recovery_at_k(ranking, labels, w.candidate_snippets, 3))
    return {
        "recall@3": _safe_mean(r3),
        "ndcg@5": _safe_mean(ndcg),
        "non_local_recovery@3": _safe_mean([v for v in nlr if not math.isnan(v)]),
    }


def load_existing_rankings(method: str) -> dict[str, list[str]]:
    path = RANKINGS_DIR / f"{method}.jsonl"
    rankings = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            rankings[row["warning_id"]] = row["ranking"]
    return rankings


# ──────────────────────────────────────────────────────────────────────────────
# BM25 scores (raw)
# ──────────────────────────────────────────────────────────────────────────────

def bm25_scores_for_warning(warning) -> dict[str, float]:
    """Return raw BM25 score per snippet_id for one warning."""
    from rank_bm25 import BM25Okapi
    query = tokenize(" ".join([
        warning.warning_message or "",
        warning.rule_id or "",
        warning.category or "",
    ]))
    docs = [tokenize(s.text) for s in warning.candidate_snippets]
    for i, d in enumerate(docs):
        if not d:
            docs[i] = [" "]
    bm25 = BM25Okapi(docs, k1=1.5, b=0.75)
    scores = bm25.get_scores(query)
    return {s.snippet_id: float(sc) for s, sc in zip(warning.candidate_snippets, scores)}


# ──────────────────────────────────────────────────────────────────────────────
# Embedding scores (raw)
# ──────────────────────────────────────────────────────────────────────────────

def _embed_all(warnings):
    """Return {warning_id: {snippet_id: cosine_score}} using MiniLM."""
    try:
        from sentence_transformers import SentenceTransformer
        model_path = "models/all-MiniLM-L6-v2"
        model = SentenceTransformer(model_path, device="cpu")
        print("  [embedding] using sentence-transformers model")
    except Exception:
        model = None
        print("  [embedding] fallback to hash embedding")

    result = {}
    for w in warnings:
        query = " ".join([w.warning_message or "", w.rule_id or "", w.category or ""])
        texts = [query] + [s.text for s in w.candidate_snippets]
        if model is not None:
            import numpy as np
            vecs = model.encode(texts, normalize_embeddings=True)
            q_vec = vecs[0]
            scores = [float(np.dot(q_vec, v)) for v in vecs[1:]]
        else:
            # hash fallback
            import hashlib
            def hv(t, dim=256):
                vec = [0.0]*dim
                for tok in TOKEN.findall(t or ""):
                    h = int(hashlib.md5(tok.lower().encode()).hexdigest(), 16)
                    idx = h % dim
                    sign = 1.0 if (h>>8)&1 else -1.0
                    vec[idx] += sign
                nm = sum(v*v for v in vec)**0.5
                return [v/nm if nm else 0 for v in vec]
            q_vec = hv(query)
            scores = [sum(a*b for a,b in zip(q_vec, hv(s.text))) for s in w.candidate_snippets]
        result[w.warning_id] = {s.snippet_id: sc for s, sc in zip(w.candidate_snippets, scores)}
    return result


# ──────────────────────────────────────────────────────────────────────────────
# E8: Random baseline 10,000 shuffles
# ──────────────────────────────────────────────────────────────────────────────

def run_E8(test_warnings):
    print("\n=== E8: Random baseline 10,000 shuffles ===")
    local_rng = random.Random(7)
    n = 10_000
    r3_runs = []
    for _ in range(n):
        r3_vals = []
        for w in test_warnings:
            ranking = [s.snippet_id for s in w.candidate_snippets]
            local_rng.shuffle(ranking)
            essentials = w.essentials()
            r3_vals.append(recall_at_k(ranking, essentials, 3))
        r3_runs.append(_safe_mean(r3_vals))
    mean_r3 = _safe_mean(r3_runs)
    std_r3 = (sum((x - mean_r3)**2 for x in r3_runs) / len(r3_runs))**0.5
    r3_sorted = sorted(r3_runs)
    ci_lo = r3_sorted[int(0.025 * (n-1))]
    ci_hi = r3_sorted[int(0.975 * (n-1))]
    result = {
        "n_shuffles": n,
        "mean_recall@3": round(mean_r3, 4),
        "std_recall@3": round(std_r3, 4),
        "ci_lo_95": round(ci_lo, 4),
        "ci_hi_95": round(ci_hi, 4),
        "2sd_lo": round(mean_r3 - 2*std_r3, 4),
        "2sd_hi": round(mean_r3 + 2*std_r3, 4),
    }
    print(f"  Recall@3 (10k shuffles): {mean_r3:.4f} ± {std_r3:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    out = OUT_DIR / "E8_random_10k.json"
    with out.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"  → {out}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# E2: Hybrid baselines (RRF + Convex + TP+BM25)
# ──────────────────────────────────────────────────────────────────────────────

def run_E2(test_warnings):
    print("\n=== E2: Hybrid retrieval baselines ===")

    # Load existing rankings
    bm25_rankings = load_existing_rankings("bm25")
    emb_rankings = load_existing_rankings("embedding_all-MiniLM-L6-v2")
    tp_rankings = load_existing_rankings("type_priority")
    lf_rankings = load_existing_rankings("local_first")

    # --- RRF (k=60) ---
    print("  Computing RRF (BM25 + Embedding) ...")
    K_RRF = 60

    def rrf_merge(ranking_a, ranking_b):
        sids = list(dict.fromkeys(ranking_a + ranking_b))
        rank_a = {s: i+1 for i, s in enumerate(ranking_a)}
        rank_b = {s: i+1 for i, s in enumerate(ranking_b)}
        scores = {s: 1/(K_RRF + rank_a.get(s, len(sids)+1)) + 1/(K_RRF + rank_b.get(s, len(sids)+1))
                  for s in sids}
        return sorted(sids, key=lambda s: -scores[s])

    rrf_rankings = {}
    for w in test_warnings:
        rrf_rankings[w.warning_id] = rrf_merge(
            bm25_rankings.get(w.warning_id, []),
            emb_rankings.get(w.warning_id, [])
        )

    # --- RRF three-way (BM25 + Embedding + TypePriority) ---
    print("  Computing RRF three-way (BM25 + Embedding + TypePriority) ...")
    rrf3_rankings = {}
    for w in test_warnings:
        sids = list(dict.fromkeys(
            bm25_rankings.get(w.warning_id, []) +
            emb_rankings.get(w.warning_id, []) +
            tp_rankings.get(w.warning_id, [])
        ))
        rank_b = {s: i+1 for i, s in enumerate(bm25_rankings.get(w.warning_id, []))}
        rank_e = {s: i+1 for i, s in enumerate(emb_rankings.get(w.warning_id, []))}
        rank_t = {s: i+1 for i, s in enumerate(tp_rankings.get(w.warning_id, []))}
        n = len(sids)
        scores = {
            s: 1/(K_RRF + rank_b.get(s, n+1))
             + 1/(K_RRF + rank_e.get(s, n+1))
             + 1/(K_RRF + rank_t.get(s, n+1))
            for s in sids
        }
        rrf3_rankings[w.warning_id] = sorted(sids, key=lambda s: -scores[s])

    # --- Convex combination with raw scores ---
    print("  Computing embedding scores for convex combination ...")
    emb_score_map = _embed_all(test_warnings)  # {wid: {sid: score}}
    print("  Computing BM25 scores ...")
    bm25_score_map = {w.warning_id: bm25_scores_for_warning(w) for w in test_warnings}

    def minmax_norm(d: dict) -> dict:
        """Normalize dict values to [0,1]."""
        if not d:
            return d
        lo, hi = min(d.values()), max(d.values())
        if hi == lo:
            return {k: 0.5 for k in d}
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}

    convex_results = {}
    for alpha in [0.3, 0.5, 0.7]:
        rankings = {}
        for w in test_warnings:
            sids = [s.snippet_id for s in w.candidate_snippets]
            b_norm = minmax_norm(bm25_score_map.get(w.warning_id, {}))
            e_norm = minmax_norm(emb_score_map.get(w.warning_id, {}))
            scores = {
                s: alpha * b_norm.get(s, 0.0) + (1 - alpha) * e_norm.get(s, 0.0)
                for s in sids
            }
            rankings[w.warning_id] = sorted(sids, key=lambda s: -scores[s])
        convex_results[alpha] = rankings

    # --- TypePriority + BM25 rerank within type buckets ---
    print("  Computing TypePriority + BM25 rerank ...")
    tp_bm25_rankings = {}
    for w in test_warnings:
        tp_order = tp_rankings.get(w.warning_id, [])
        # Group by type
        by_id = {s.snippet_id: s for s in w.candidate_snippets}
        b_scores = bm25_score_map.get(w.warning_id, {})
        # Assign "type rank" based on position in tp_order then rerank within same type group
        # Actually: keep tp ordering but within same-type group, sort by BM25 desc
        type_of = {s.snippet_id: s.type for s in w.candidate_snippets}
        # Build type groups preserving tp order
        seen_types = {}
        for i, sid in enumerate(tp_order):
            t = type_of.get(sid, "unknown")
            seen_types.setdefault(t, []).append((i, sid))
        # Sort within each type group by BM25 score desc
        final_order = {}
        for t, items in seen_types.items():
            sorted_items = sorted(items, key=lambda x: -b_scores.get(x[1], 0.0))
            for rank_in_type, (orig_pos, sid) in enumerate(sorted_items):
                final_order[sid] = (orig_pos, rank_in_type)  # (type_group_pos, within_type_rank)
        # Sort: primary by first-occurrence position of type group, secondary by within-type rank
        type_first_pos = {}
        for i, sid in enumerate(tp_order):
            t = type_of.get(sid, "unknown")
            if t not in type_first_pos:
                type_first_pos[t] = i
        tp_bm25_rankings[w.warning_id] = sorted(
            [s.snippet_id for s in w.candidate_snippets],
            key=lambda sid: (type_first_pos.get(type_of.get(sid, ""), 999), final_order.get(sid, (999, 999))[1])
        )

    # --- Evaluate all hybrid methods ---
    results = []
    methods = {
        "rrf_bm25_emb": rrf_rankings,
        "rrf_bm25_emb_tp": rrf3_rankings,
        "convex_0.3": convex_results[0.3],
        "convex_0.5": convex_results[0.5],
        "convex_0.7": convex_results[0.7],
        "tp_bm25_rerank": tp_bm25_rankings,
        # Baselines for comparison
        "bm25": bm25_rankings,
        "embedding": emb_rankings,
        "type_priority": tp_rankings,
        "local_first": lf_rankings,
    }
    for name, rankings in methods.items():
        m = _eval_metrics(test_warnings, rankings)
        m["method"] = name
        results.append(m)
        print(f"  {name:30s}  R@3={m['recall@3']:.4f}  nDCG@5={m['ndcg@5']:.4f}  NLR@3={m['non_local_recovery@3']:.4f}")

    out = OUT_DIR / "E2_hybrid_baselines.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "recall@3", "ndcg@5", "non_local_recovery@3"])
        w.writeheader()
        w.writerows(results)
    print(f"  → {out}")
    return results, emb_score_map, bm25_score_map


# ──────────────────────────────────────────────────────────────────────────────
# E3: Learning-to-Rank baseline
# ──────────────────────────────────────────────────────────────────────────────

def compute_features(warning, snippet, bm25_score, emb_score):
    """16-dim feature vector for (warning, snippet) pair."""
    # Type one-hot (11 dims)
    type_vec = [1 if snippet.type == t else 0 for t in SNIPPET_TYPES]
    # BM25 score (1 dim)
    # Embedding cosine (1 dim)
    # Distance (1 dim, normalized)
    mid = (snippet.line_start + snippet.line_end) / 2
    distance = abs(mid - warning.line) if warning.line > 0 else 0
    dist_norm = min(distance / 100.0, 10.0)  # clip at 10
    # Log token count (1 dim)
    log_tok = math.log(1 + snippet.token_count)
    # Is local (1 dim)
    is_local = 1 if snippet.type in LOCAL_SNIPPET_TYPES else 0
    return type_vec + [bm25_score, emb_score, dist_norm, log_tok, is_local]


def run_E3(all_warnings, test_warnings):
    print("\n=== E3: Learning-to-Rank baseline ===")

    train_warnings = warnings_by_split(all_warnings, "train")

    # Compute embedding and BM25 scores for train + test
    print("  Computing embedding scores for train split ...")
    train_emb = _embed_all(train_warnings)
    print("  Computing embedding scores for test split ...")
    test_emb = _embed_all(test_warnings)
    print("  Computing BM25 scores ...")
    train_bm25 = {w.warning_id: bm25_scores_for_warning(w) for w in train_warnings}
    test_bm25 = {w.warning_id: bm25_scores_for_warning(w) for w in test_warnings}

    rel_map = {"essential": 2, "helpful": 1, "irrelevant": 0}

    # Build pairwise training data
    print("  Building pairwise training data ...")
    X_train, y_train = [], []
    for w in train_warnings:
        labels_map = {lbl.snippet_id: lbl.relevance for lbl in w.labels}
        snippets = w.candidate_snippets
        bm25_sc = train_bm25.get(w.warning_id, {})
        emb_sc = train_emb.get(w.warning_id, {})
        for i in range(len(snippets)):
            for j in range(len(snippets)):
                if i == j:
                    continue
                si, sj = snippets[i], snippets[j]
                rel_i = rel_map.get(labels_map.get(si.snippet_id, "irrelevant"), 0)
                rel_j = rel_map.get(labels_map.get(sj.snippet_id, "irrelevant"), 0)
                if rel_i == rel_j:
                    continue
                fi = compute_features(w, si, bm25_sc.get(si.snippet_id, 0.0), emb_sc.get(si.snippet_id, 0.0))
                fj = compute_features(w, sj, bm25_sc.get(sj.snippet_id, 0.0), emb_sc.get(sj.snippet_id, 0.0))
                diff = [a - b for a, b in zip(fi, fj)]
                X_train.append(diff)
                y_train.append(1 if rel_i > rel_j else 0)

    print(f"  Training pairs: {len(X_train)} (positive={sum(y_train)}, negative={len(y_train)-sum(y_train)})")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        X = np.array(X_train, dtype=float)
        y = np.array(y_train, dtype=int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)
        clf.fit(X_scaled, y)
        print(f"  Model trained. Coefficients: {clf.coef_.shape}")

        # Score each snippet for test warnings (pointwise: score = f(si) - f(zero_vec))
        # Equivalent to: rank by clf's decision function on each snippet's features vs zero
        # We use a pointwise approach: score(si) = w·f(si)
        w_vec = clf.coef_[0]  # (16,)

        ltr_rankings = {}
        for w in test_warnings:
            bm25_sc = test_bm25.get(w.warning_id, {})
            emb_sc = test_emb.get(w.warning_id, {})
            scores = []
            for s in w.candidate_snippets:
                feat = compute_features(w, s, bm25_sc.get(s.snippet_id, 0.0), emb_sc.get(s.snippet_id, 0.0))
                feat_scaled = scaler.transform([feat])[0]
                score = float(np.dot(w_vec, feat_scaled))
                scores.append((s.snippet_id, score))
            scores.sort(key=lambda x: -x[1])
            ltr_rankings[w.warning_id] = [sid for sid, _ in scores]

        m = _eval_metrics(test_warnings, ltr_rankings)
        m["method"] = "ltr_logistic_pairwise"
        print(f"  LTR (LogisticRegression pairwise):  R@3={m['recall@3']:.4f}  nDCG@5={m['ndcg@5']:.4f}  NLR@3={m['non_local_recovery@3']:.4f}")

        # Feature importance
        feat_names = list(SNIPPET_TYPES) + ["bm25_score", "emb_cosine", "dist_norm", "log_tokens", "is_local"]
        coef_pairs = sorted(zip(feat_names, w_vec), key=lambda x: -abs(x[1]))
        print("  Top features by |coeff|:")
        for name, coef in coef_pairs[:8]:
            print(f"    {name:35s}  {coef:+.4f}")

        results = [m]
        out = OUT_DIR / "E3_ltr_baseline.csv"
        with out.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["method", "recall@3", "ndcg@5", "non_local_recovery@3"])
            wr.writeheader()
            wr.writerows(results)

        # Save feature importance
        fi_out = OUT_DIR / "E3_feature_importance.csv"
        with fi_out.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["feature", "coefficient"])
            wr.writeheader()
            wr.writerows({"feature": n, "coefficient": round(c, 5)} for n, c in zip(feat_names, w_vec))
        print(f"  → {out}, {fi_out}")
        return results, ltr_rankings

    except ImportError:
        print("  scikit-learn not available, skipping E3")
        return [], {}


# ──────────────────────────────────────────────────────────────────────────────
# E4: Type-priority failure analysis
# ──────────────────────────────────────────────────────────────────────────────

def run_E4(test_warnings):
    print("\n=== E4: Type-priority failure analysis ===")
    tp_rankings = load_existing_rankings("type_priority")

    failure_rows = []
    for w in test_warnings:
        ranking = tp_rankings.get(w.warning_id, [])
        labels_map = {lbl.snippet_id: lbl.relevance for lbl in w.labels}
        essentials = w.essentials()
        r3 = recall_at_k(ranking, essentials, 3)
        by_id = {s.snippet_id: s for s in w.candidate_snippets}

        missed = essentials - set(ranking[:3])
        for sid in missed:
            snip = by_id.get(sid)
            if snip is None:
                continue
            # What rank does it actually appear at?
            actual_rank = ranking.index(sid) + 1 if sid in ranking else -1
            # What types are in top-3?
            top3_types = [by_id[s].type for s in ranking[:3] if s in by_id]
            failure_rows.append({
                "warning_id": w.warning_id,
                "category": w.category,
                "rule_id": w.rule_id,
                "recall@3": round(r3, 4),
                "n_essentials": len(essentials),
                "n_missed": len(missed),
                "missed_snippet_id": sid,
                "missed_snippet_type": snip.type,
                "missed_token_count": snip.token_count,
                "actual_rank": actual_rank,
                "top3_types": "|".join(top3_types),
                "note": "non_local" if snip.type not in LOCAL_SNIPPET_TYPES else "local",
            })

    # Summary stats
    total_warnings = len(test_warnings)
    failing = len(set(r["warning_id"] for r in failure_rows))
    print(f"  Warnings with R@3 < 1.0: {failing}/{total_warnings} ({100*failing/total_warnings:.1f}%)")
    if failure_rows:
        # By missed type
        type_counts = defaultdict(int)
        for r in failure_rows:
            type_counts[r["missed_snippet_type"]] += 1
        print("  Missed snippet types:")
        for t, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"    {t:35s}  {cnt}")
        # By category
        cat_counts = defaultdict(int)
        for r in failure_rows:
            cat_counts[r["category"]] += 1
        print("  Failures by category:")
        for c, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"    {c:30s}  {cnt}")

    out = OUT_DIR / "E4_tp_failure_analysis.csv"
    with out.open("w", newline="") as f:
        if failure_rows:
            wr = csv.DictWriter(f, fieldnames=list(failure_rows[0].keys()))
            wr.writeheader()
            wr.writerows(failure_rows)
    print(f"  → {out}  ({len(failure_rows)} missed essential snippets)")
    return failure_rows


# ──────────────────────────────────────────────────────────────────────────────
# E5: Hard-negative sensitivity
# ──────────────────────────────────────────────────────────────────────────────

def run_E5(test_warnings, all_warnings):
    print("\n=== E5: Hard-negative sensitivity analysis ===")

    # Load hard-neg pool from existing dataset
    hn_warnings_all = load_warnings(HARDNEG_PATH)
    hn_test = warnings_by_split(hn_warnings_all, "test")
    hn_by_id = {w.warning_id: w for w in hn_test}

    # Extract the pre-built hard-negative snippets per warning
    hn_pool = {}  # warning_id → list of hard-negative CandidateSnippets
    for w in hn_test:
        hn_pool[w.warning_id] = [s for s in w.candidate_snippets if s.snippet_id.startswith("hn")]

    # Prepare rankers
    train_warnings = warnings_by_split(all_warnings, "train")
    val_warnings = warnings_by_split(all_warnings, "validation")
    tp_ranker = TypePriorityRanker()
    tp_ranker.fit(val_warnings)

    from saw_bench_cs.baselines.bm25 import BM25Ranker
    from saw_bench_cs.baselines.local_first import LocalFirstRanker
    bm25_ranker = BM25Ranker()
    lf_ranker = LocalFirstRanker()

    # Use embedding ranker with local model
    try:
        from saw_bench_cs.baselines.embedding import EmbeddingRanker
        emb_ranker = EmbeddingRanker(model_name="models/all-MiniLM-L6-v2")
    except Exception:
        emb_ranker = None

    ratios = [0.0, 0.25, 0.5, 1.0]
    results = []

    for ratio in ratios:
        print(f"  Ratio = {ratio:.2f} ...")
        # Build augmented warnings for this ratio
        augmented = []
        for w in test_warnings:
            import copy
            # Create a copy-like object with modified candidate_snippets
            hn_snippets = hn_pool.get(w.warning_id, [])
            n_inject = int(math.ceil(len(w.candidate_snippets) * ratio))
            n_inject = min(n_inject, len(hn_snippets))
            injected = hn_snippets[:n_inject]

            # Build augmented warning by modifying in-place representation
            # We'll use a simple wrapper
            class AugWarning:
                pass
            aw = AugWarning()
            aw.warning_id = w.warning_id
            aw.project = w.project
            aw.rule_id = w.rule_id
            aw.category = w.category
            aw.file = w.file
            aw.line = w.line
            aw.warning_message = w.warning_message
            aw.warning_context = w.warning_context
            aw.candidate_snippets = list(w.candidate_snippets) + list(injected)
            aw.labels = w.labels + [
                type('Label', (), {'snippet_id': s.snippet_id, 'relevance': 'irrelevant'})()
                for s in injected
            ]
            aw.split = w.split

            def essentials(self=aw):
                return {lbl.snippet_id for lbl in self.labels if lbl.relevance == "essential"}
            aw.essentials = essentials

            def useful(self=aw):
                return {lbl.snippet_id for lbl in self.labels if lbl.relevance in ("essential", "helpful")}
            aw.useful = useful

            augmented.append(aw)

        # Run each method
        for method_name, ranker in [
            ("local_first", lf_ranker),
            ("bm25", bm25_ranker),
            ("embedding", emb_ranker),
            ("type_priority", tp_ranker),
        ]:
            if ranker is None:
                continue
            r3_vals, nlr_vals = [], []
            for aw in augmented:
                try:
                    ranking = ranker.rank(aw)
                except Exception:
                    ranking = [s.snippet_id for s in aw.candidate_snippets]
                essentials = aw.essentials()
                labels_map = {lbl.snippet_id: lbl.relevance for lbl in aw.labels}
                r3_vals.append(recall_at_k(ranking, essentials, 3))
                nlr_vals.append(non_local_recovery_at_k(ranking, labels_map, aw.candidate_snippets, 3))

            row = {
                "ratio": ratio,
                "n_inject_avg": round(ratio * _safe_mean([len(w.candidate_snippets) for w in test_warnings]), 2),
                "method": method_name,
                "recall@3": round(_safe_mean(r3_vals), 4),
                "non_local_recovery@3": round(_safe_mean([v for v in nlr_vals if not math.isnan(v)]), 4),
            }
            results.append(row)
            print(f"    {method_name:30s}  R@3={row['recall@3']:.4f}  NLR@3={row['non_local_recovery@3']:.4f}")

    out = OUT_DIR / "E5_hardneg_sensitivity.csv"
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["ratio", "n_inject_avg", "method", "recall@3", "non_local_recovery@3"])
        wr.writeheader()
        wr.writerows(results)
    print(f"  → {out}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# E6: Candidate generation coverage analysis
# ──────────────────────────────────────────────────────────────────────────────

def run_E6(test_warnings):
    print("\n=== E6: Candidate generation coverage analysis ===")

    rows = []
    total_essentials = 0
    total_nonlocal_essentials = 0
    upstream_misses = 0
    nonlocal_upstream_misses = 0

    for w in test_warnings:
        pool_ids = {s.snippet_id for s in w.candidate_snippets}
        labels_map = {lbl.snippet_id: lbl.relevance for lbl in w.labels}
        by_id = {s.snippet_id: s for s in w.candidate_snippets}

        essentials = w.essentials()
        for sid in essentials:
            total_essentials += 1
            in_pool = sid in pool_ids
            snip = by_id.get(sid)
            snip_type = snip.type if snip else "MISSING"
            is_nonlocal = snip_type not in LOCAL_SNIPPET_TYPES if snip else True

            if is_nonlocal:
                total_nonlocal_essentials += 1
            if not in_pool:
                upstream_misses += 1
                if is_nonlocal:
                    nonlocal_upstream_misses += 1

            rows.append({
                "warning_id": w.warning_id,
                "category": w.category,
                "rule_id": w.rule_id,
                "snippet_id": sid,
                "snippet_type": snip_type,
                "is_nonlocal": is_nonlocal,
                "in_pool": in_pool,
            })

    print(f"  Total essential snippets: {total_essentials}")
    print(f"  Upstream misses (essential not in pool): {upstream_misses} ({100*upstream_misses/max(total_essentials,1):.1f}%)")
    print(f"  Non-local essential snippets: {total_nonlocal_essentials}")
    print(f"  Non-local upstream misses: {nonlocal_upstream_misses} ({100*nonlocal_upstream_misses/max(total_nonlocal_essentials,1):.1f}%)")

    # Per-type analysis
    type_totals = defaultdict(int)
    type_misses = defaultdict(int)
    for row in rows:
        type_totals[row["snippet_type"]] += 1
        if not row["in_pool"]:
            type_misses[row["snippet_type"]] += 1
    print("  Coverage by type:")
    for t in sorted(type_totals.keys()):
        total = type_totals[t]
        miss = type_misses[t]
        print(f"    {t:35s}  {total-miss}/{total} in pool ({100*(total-miss)/total:.0f}%)")

    out = OUT_DIR / "E6_coverage_analysis.csv"
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    summary = {
        "total_essential": total_essentials,
        "upstream_misses": upstream_misses,
        "upstream_miss_rate": round(upstream_misses / max(total_essentials, 1), 4),
        "total_nonlocal_essential": total_nonlocal_essentials,
        "nonlocal_upstream_misses": nonlocal_upstream_misses,
        "nonlocal_upstream_miss_rate": round(nonlocal_upstream_misses / max(total_nonlocal_essentials, 1), 4),
    }
    with (OUT_DIR / "E6_coverage_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → {out}")
    return rows, summary


# ──────────────────────────────────────────────────────────────────────────────
# E7: Per-category bootstrap CIs for non-local R@3
# ──────────────────────────────────────────────────────────────────────────────

def run_E7(test_warnings):
    print("\n=== E7: Per-category bootstrap CIs for non-local R@3 ===")

    methods = {
        "bm25": load_existing_rankings("bm25"),
        "embedding": load_existing_rankings("embedding_all-MiniLM-L6-v2"),
        "local_first": load_existing_rankings("local_first"),
        "type_priority": load_existing_rankings("type_priority"),
    }

    # Group by category
    by_cat = defaultdict(list)
    for w in test_warnings:
        by_cat[w.category].append(w)

    N_BOOT = 1000
    local_rng = random.Random(7)
    rows = []

    for cat, cat_warnings in sorted(by_cat.items()):
        n = len(cat_warnings)
        for method_name, rankings in methods.items():
            # Point estimates
            nlr_vals = []
            r3_vals = []
            for w in cat_warnings:
                ranking = rankings.get(w.warning_id, [])
                labels_map = {lbl.snippet_id: lbl.relevance for lbl in w.labels}
                essentials = w.essentials()
                nlr = non_local_recovery_at_k(ranking, labels_map, w.candidate_snippets, 3)
                nlr_vals.append(nlr)
                r3_vals.append(recall_at_k(ranking, essentials, 3))

            # Bootstrap
            clean_nlr = [v for v in nlr_vals if not math.isnan(v)]
            if len(clean_nlr) < 2:
                # Not enough data for CI
                rows.append({
                    "category": cat,
                    "n_warnings": n,
                    "method": method_name,
                    "recall@3_mean": round(_safe_mean(r3_vals), 4),
                    "nonlocal_r3_mean": round(_safe_mean(clean_nlr), 4) if clean_nlr else "nan",
                    "nonlocal_r3_ci_lo": "nan",
                    "nonlocal_r3_ci_hi": "nan",
                    "n_with_nonlocal_essential": len(clean_nlr),
                })
                continue

            boot_means = []
            for _ in range(N_BOOT):
                sample = [clean_nlr[local_rng.randrange(len(clean_nlr))] for _ in clean_nlr]
                boot_means.append(_safe_mean(sample))
            boot_means.sort()
            ci_lo = boot_means[int(0.025 * (N_BOOT - 1))]
            ci_hi = boot_means[int(0.975 * (N_BOOT - 1))]

            rows.append({
                "category": cat,
                "n_warnings": n,
                "method": method_name,
                "recall@3_mean": round(_safe_mean(r3_vals), 4),
                "nonlocal_r3_mean": round(_safe_mean(clean_nlr), 4),
                "nonlocal_r3_ci_lo": round(ci_lo, 4),
                "nonlocal_r3_ci_hi": round(ci_hi, 4),
                "n_with_nonlocal_essential": len(clean_nlr),
            })

    # Print summary table
    print(f"  {'Category':30s} {'n':>4} {'Method':20s} {'NLR@3':>6} {'CI':>16}")
    for r in rows:
        if r["method"] != "type_priority":
            continue
        ci = f"[{r['nonlocal_r3_ci_lo']}, {r['nonlocal_r3_ci_hi']}]"
        print(f"  {r['category']:30s} {r['n_warnings']:>4} {r['method']:20s} {str(r['nonlocal_r3_mean']):>6} {ci:>16}")

    out = OUT_DIR / "E7_per_category_bootstrap.csv"
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"  → {out}")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading dataset ...")
    all_warnings = load_warnings(DATA_PATH)
    test_warnings = warnings_by_split(all_warnings, "test")
    print(f"Test split: {len(test_warnings)} warnings\n")

    os.chdir(Path(__file__).parent.parent)  # cd to project root

    # E8 first (fast)
    run_E8(test_warnings)

    # E4 (fast, uses existing rankings)
    run_E4(test_warnings)

    # E6 (fast)
    run_E6(test_warnings)

    # E7 (medium)
    run_E7(test_warnings)

    # E2 (slow: needs embedding computation)
    e2_results, emb_scores, bm25_scores = run_E2(test_warnings)

    # E3 (slow: needs train+test embedding)
    run_E3(all_warnings, test_warnings)

    # E5 (medium: needs rankers)
    run_E5(test_warnings, all_warnings)

    print("\n✓ All extra experiments complete. Results in:", OUT_DIR)


if __name__ == "__main__":
    main()
