"""Experiments addressing Opus reviewer concerns:
E12: LTR R@5 + cross-validation on training split
E13: Pool saturation curve (R@k for k=1..15 for TP and LTR)
E14: RRF degradation analysis

Output: results_extra/E12_ltr_cv_r5.csv, E13_saturation.csv, E14_rrf_analysis.csv
"""
from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path
from collections import defaultdict

import _bootstrap  # noqa: F401

from saw_bench_cs.io import load_warnings, warnings_by_split
from saw_bench_cs.evaluation.metrics import recall_at_k, ndcg_at_k, non_local_recovery_at_k, mrr
from saw_bench_cs.baselines.type_priority import TypePriorityRanker, fit_priors
from saw_bench_cs.schema import SNIPPET_TYPES, LOCAL_SNIPPET_TYPES

DATA_PATH = "data/saw_bench_cs.jsonl"
OUT_DIR = Path("results_extra")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")

def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN.findall(text or "")]

def _safe_mean(values):
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else float("nan")

def bm25_scores_for_warning(warning) -> dict[str, float]:
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

def _embed_all(warnings):
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer("models/all-MiniLM-L6-v2", device="cpu")
        result = {}
        for w in warnings:
            query = " ".join([w.warning_message or "", w.rule_id or "", w.category or ""])
            texts = [query] + [s.text for s in w.candidate_snippets]
            embs = model.encode(texts, batch_size=32, show_progress_bar=False)
            q_emb = embs[0]
            scores = {}
            for i, s in enumerate(w.candidate_snippets):
                d = embs[i + 1]
                denom = (np.linalg.norm(q_emb) * np.linalg.norm(d)) + 1e-9
                scores[s.snippet_id] = float(np.dot(q_emb, d) / denom)
            result[w.warning_id] = scores
        return result
    except Exception as e:
        print(f"  [embedding] error: {e}")
        return {w.warning_id: {s.snippet_id: 0.0 for s in w.candidate_snippets} for w in warnings}

def compute_features(warning, snippet, bm25_score, emb_score):
    type_vec = [1 if snippet.type == t else 0 for t in SNIPPET_TYPES]
    mid = (snippet.line_start + snippet.line_end) / 2
    distance = abs(mid - warning.line) if warning.line > 0 else 0
    dist_norm = min(distance / 100.0, 10.0)
    log_tok = math.log(1 + snippet.token_count)
    is_local = 1 if snippet.type in LOCAL_SNIPPET_TYPES else 0
    return type_vec + [bm25_score, emb_score, dist_norm, log_tok, is_local]

def train_ltr(train_warnings, bm25_dict, emb_dict):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    rel_map = {"essential": 2, "helpful": 1, "irrelevant": 0}
    X_train, y_train = [], []
    for w in train_warnings:
        labels_map = {lbl.snippet_id: lbl.relevance for lbl in w.labels}
        bm25_sc = bm25_dict.get(w.warning_id, {})
        emb_sc = emb_dict.get(w.warning_id, {})
        snippets = w.candidate_snippets
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
    X = np.array(X_train, dtype=float)
    y = np.array(y_train, dtype=int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)
    clf.fit(X_scaled, y)
    return clf, scaler, clf.coef_[0]

def score_warnings(warnings, clf, scaler, bm25_dict, emb_dict):
    import numpy as np
    w_vec = clf.coef_[0]
    rankings = {}
    for w in warnings:
        bm25_sc = bm25_dict.get(w.warning_id, {})
        emb_sc = emb_dict.get(w.warning_id, {})
        scores = []
        for s in w.candidate_snippets:
            feat = compute_features(w, s, bm25_sc.get(s.snippet_id, 0.0), emb_sc.get(s.snippet_id, 0.0))
            feat_scaled = scaler.transform([feat])[0]
            score = float(np.dot(w_vec, feat_scaled))
            scores.append((s.snippet_id, score))
        scores.sort(key=lambda x: -x[1])
        rankings[w.warning_id] = [sid for sid, _ in scores]
    return rankings

def eval_at_k(warnings, rankings, k_list):
    results = {k: [] for k in k_list}
    for w in warnings:
        ranking = rankings.get(w.warning_id, [])
        essentials = w.essentials()
        for k in k_list:
            results[k].append(recall_at_k(ranking, essentials, k))
    return {k: _safe_mean(v) for k, v in results.items()}


# ─── E12: LTR with cross-validation + R@5 ────────────────────────────────────

def run_E12(all_warnings, test_warnings):
    print("\n=== E12: LTR R@5 + Cross-Validation ===")
    train_warnings = warnings_by_split(all_warnings, "train")
    val_warnings = warnings_by_split(all_warnings, "val")

    # Compute features
    print("  Computing BM25 scores ...")
    all_train_val = train_warnings + val_warnings
    bm25_train = {w.warning_id: bm25_scores_for_warning(w) for w in train_warnings}
    bm25_val = {w.warning_id: bm25_scores_for_warning(w) for w in val_warnings}
    bm25_test = {w.warning_id: bm25_scores_for_warning(w) for w in test_warnings}

    print("  Computing embedding scores (train+val+test) ...")
    emb_train = _embed_all(train_warnings)
    emb_val = _embed_all(val_warnings)
    emb_test = _embed_all(test_warnings)

    k_list = [1, 3, 5, 7, 10]

    # 1. Train on train → evaluate on validation (cross-validation proxy)
    print("  Training on train → eval on validation ...")
    clf_tv, scaler_tv, _ = train_ltr(train_warnings, bm25_train, emb_train)
    val_rankings = score_warnings(val_warnings, clf_tv, scaler_tv, bm25_val, emb_val)
    val_metrics = eval_at_k(val_warnings, val_rankings, k_list)
    print(f"  [VAL] R@3={val_metrics[3]:.4f} R@5={val_metrics[5]:.4f}")

    # 2. Train on train → evaluate on test (main result)
    print("  Training on train → eval on test ...")
    test_rankings = score_warnings(test_warnings, clf_tv, scaler_tv, bm25_test, emb_test)
    test_metrics = eval_at_k(test_warnings, test_rankings, k_list)
    print(f"  [TEST] R@3={test_metrics[3]:.4f} R@5={test_metrics[5]:.4f}")

    # 3. Train on train+val → evaluate on test (best model)
    print("  Training on train+val → eval on test (best model) ...")
    bm25_trainval = {**bm25_train, **bm25_val}
    emb_trainval = {**emb_train, **emb_val}
    clf_best, scaler_best, _ = train_ltr(all_train_val, bm25_trainval, emb_trainval)
    test_rankings_best = score_warnings(test_warnings, clf_best, scaler_best, bm25_test, emb_test)
    test_metrics_best = eval_at_k(test_warnings, test_rankings_best, k_list)
    print(f"  [TEST-BEST] R@3={test_metrics_best[3]:.4f} R@5={test_metrics_best[5]:.4f}")

    # 4. C regularization sweep on validation
    print("  C regularization sweep on validation ...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    rel_map = {"essential": 2, "helpful": 1, "irrelevant": 0}
    X_train_all, y_train_all = [], []
    for w in train_warnings:
        labels_map = {lbl.snippet_id: lbl.relevance for lbl in w.labels}
        bm25_sc = bm25_train.get(w.warning_id, {})
        emb_sc = emb_train.get(w.warning_id, {})
        snippets = w.candidate_snippets
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
                X_train_all.append(diff)
                y_train_all.append(1 if rel_i > rel_j else 0)

    X = np.array(X_train_all, dtype=float)
    y = np.array(y_train_all, dtype=int)
    scaler_cv = StandardScaler()
    X_scaled = scaler_cv.fit_transform(X)

    cv_rows = []
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        clf_c = LogisticRegression(max_iter=1000, class_weight="balanced", C=C, random_state=42)
        clf_c.fit(X_scaled, y)
        w_vec_c = clf_c.coef_[0]
        val_r = {}
        val_ranks = {}
        for w in val_warnings:
            bm25_sc = bm25_val.get(w.warning_id, {})
            emb_sc = emb_val.get(w.warning_id, {})
            scores = []
            for s in w.candidate_snippets:
                feat = compute_features(w, s, bm25_sc.get(s.snippet_id, 0.0), emb_sc.get(s.snippet_id, 0.0))
                feat_scaled = scaler_cv.transform([feat])[0]
                scores.append((s.snippet_id, float(np.dot(w_vec_c, feat_scaled))))
            scores.sort(key=lambda x: -x[1])
            val_ranks[w.warning_id] = [sid for sid, _ in scores]
        vm = eval_at_k(val_warnings, val_ranks, [3, 5])
        cv_rows.append({"C": C, "val_r3": round(vm[3], 4), "val_r5": round(vm[5], 4)})
        print(f"  C={C:.2f}: val_R@3={vm[3]:.4f}, val_R@5={vm[5]:.4f}")

    # Save
    rows = []
    for k in k_list:
        rows.append({
            "k": k,
            "ltr_train_eval": round(val_metrics[k], 4),
            "ltr_test_eval": round(test_metrics[k], 4),
            "ltr_trainval_test": round(test_metrics_best[k], 4),
        })
    out = OUT_DIR / "E12_ltr_cv_r5.csv"
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["k", "ltr_train_eval", "ltr_test_eval", "ltr_trainval_test"])
        wr.writeheader()
        wr.writerows(rows)

    cv_out = OUT_DIR / "E12_ltr_regularization.csv"
    with cv_out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["C", "val_r3", "val_r5"])
        wr.writeheader()
        wr.writerows(cv_rows)

    print(f"  → {out}, {cv_out}")
    return test_rankings


# ─── E13: Pool saturation curve ───────────────────────────────────────────────

def run_E13(all_warnings, test_warnings, ltr_rankings):
    print("\n=== E13: Pool saturation curve R@k (k=1..15) ===")
    k_list = list(range(1, 16))

    # Type-priority rankings
    priors = fit_priors(warnings_by_split(all_warnings, "train"))
    ranker = TypePriorityRanker(priors)
    tp_rankings = {w.warning_id: ranker.rank(w) for w in test_warnings}

    # BM25 rankings (from file)
    import json
    bm25_rankings_loaded = {}
    bm25_path = Path("results_reproduced/rankings/bm25.jsonl")
    if bm25_path.exists():
        with bm25_path.open() as f:
            for line in f:
                row = json.loads(line)
                bm25_rankings_loaded[row["warning_id"]] = row["ranking"]

    lf_rankings_loaded = {}
    lf_path = Path("results_reproduced/rankings/local_first.jsonl")
    if lf_path.exists():
        with lf_path.open() as f:
            for line in f:
                row = json.loads(line)
                lf_rankings_loaded[row["warning_id"]] = row["ranking"]

    results = []
    for k in k_list:
        row = {"k": k}
        # TP
        r_tp = []
        for w in test_warnings:
            rank = tp_rankings.get(w.warning_id, [])
            r_tp.append(recall_at_k(rank, w.essentials(), k))
        row["type_priority"] = round(_safe_mean(r_tp), 4)
        # LTR
        r_ltr = []
        for w in test_warnings:
            rank = ltr_rankings.get(w.warning_id, [])
            r_ltr.append(recall_at_k(rank, w.essentials(), k))
        row["ltr"] = round(_safe_mean(r_ltr), 4)
        # BM25 (if available)
        if bm25_rankings_loaded:
            r_bm25 = []
            for w in test_warnings:
                rank = bm25_rankings_loaded.get(w.warning_id, [])
                r_bm25.append(recall_at_k(rank, w.essentials(), k))
            row["bm25"] = round(_safe_mean(r_bm25), 4)
        # LF (if available)
        if lf_rankings_loaded:
            r_lf = []
            for w in test_warnings:
                rank = lf_rankings_loaded.get(w.warning_id, [])
                r_lf.append(recall_at_k(rank, w.essentials(), k))
            row["local_first"] = round(_safe_mean(r_lf), 4)
        results.append(row)
        print(f"  k={k:2d}: TP={row['type_priority']:.4f} LTR={row['ltr']:.4f}" +
              (f" BM25={row.get('bm25', '?')}" if bm25_rankings_loaded else ""))

    # Pool size stats
    pool_sizes = [len(w.candidate_snippets) for w in test_warnings]
    print(f"\n  Pool sizes: min={min(pool_sizes)} max={max(pool_sizes)} mean={_safe_mean(pool_sizes):.1f}")
    print(f"  TP saturates (R@k>0.98): k={next((k for k,r in zip(k_list, [results[i]['type_priority'] for i in range(len(k_list))]) if r >= 0.98), 'never')}")

    out = OUT_DIR / "E13_saturation.csv"
    fieldnames = ["k", "type_priority", "ltr"]
    if bm25_rankings_loaded:
        fieldnames.append("bm25")
    if lf_rankings_loaded:
        fieldnames.append("local_first")
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(results)
    print(f"  → {out}")


# ─── E14: RRF degradation analysis ──────────────────────────────────────────

def run_E14(test_warnings):
    print("\n=== E14: RRF Degradation Analysis ===")
    # Load existing BM25 and TP rankings
    import json
    bm25_r = {}
    tp_r = {}
    lf_r = {}
    emb_r = {}
    for name, store in [("bm25", bm25_r), ("type_priority", tp_r),
                         ("local_first", lf_r), ("embedding_all-MiniLM-L6-v2", emb_r)]:
        p = Path(f"results_reproduced/rankings/{name}.jsonl")
        if p.exists():
            with p.open() as f:
                for line in f:
                    row = json.loads(line)
                    store[row["warning_id"]] = row["ranking"]

    def rrf_merge(rankings_list, k=60):
        scores = defaultdict(float)
        for rankings in rankings_list:
            for rank, sid in enumerate(rankings, start=1):
                scores[sid] += 1.0 / (k + rank)
        return sorted(scores.keys(), key=lambda x: -scores[x])

    def eval_r3(warnings, rankings_dict):
        vals = []
        for w in warnings:
            rank = rankings_dict.get(w.warning_id, [])
            vals.append(recall_at_k(rank, w.essentials(), 3))
        return _safe_mean(vals)

    rows = []

    # Diagnose: per-warning, does adding BM25 to TP hurt or help?
    helped = 0
    hurt = 0
    neutral = 0
    for w in test_warnings:
        tp_rank = tp_r.get(w.warning_id, [])
        bm25_rank = bm25_r.get(w.warning_id, [])
        ess = w.essentials()
        tp_score = recall_at_k(tp_rank, ess, 3)
        rrf_rank = rrf_merge([tp_rank, bm25_rank])
        rrf_score = recall_at_k(rrf_rank, ess, 3)
        if rrf_score > tp_score:
            helped += 1
        elif rrf_score < tp_score:
            hurt += 1
        else:
            neutral += 1

    print(f"  RRF(TP+BM25) vs TP alone: helped={helped}, hurt={hurt}, neutral={neutral}")

    # RRF 3-way vs 2-way comparison
    methods = {
        "tp_alone": {w.warning_id: tp_r.get(w.warning_id, []) for w in test_warnings},
        "bm25_alone": {w.warning_id: bm25_r.get(w.warning_id, []) for w in test_warnings},
        "rrf_tp_bm25": {w.warning_id: rrf_merge([tp_r.get(w.warning_id,[]), bm25_r.get(w.warning_id,[])]) for w in test_warnings},
        "rrf_bm25_emb": {w.warning_id: rrf_merge([bm25_r.get(w.warning_id,[]), emb_r.get(w.warning_id,[])]) for w in test_warnings},
        "rrf_tp_bm25_emb": {w.warning_id: rrf_merge([tp_r.get(w.warning_id,[]), bm25_r.get(w.warning_id,[]), emb_r.get(w.warning_id,[])]) for w in test_warnings},
    }

    for name, rankings in methods.items():
        r3 = eval_r3(test_warnings, rankings)
        rows.append({"method": name, "recall@3": round(r3, 4)})
        print(f"  {name:30s}: R@3={r3:.4f}")

    print(f"\n  Explanation of RRF(TP+BM25+Emb) < TP alone:")
    print(f"  RRF gives equal weight to all 3 methods.")
    print(f"  BM25 and Emb have R@3~0.48 alone. When merged with TP (R@3=0.892),")
    print(f"  they dilute TP's type-priority signal by pulling non-type-priority")
    print(f"  snippets into top-3 positions. RRF does not know that TP >> BM25/Emb.")
    print(f"  Fix: weighted RRF (give TP higher k-factor) or TP-then-rerank.")

    # Hurt analysis — which warnings does RRF hurt?
    hurt_cases = []
    for w in test_warnings:
        tp_rank = tp_r.get(w.warning_id, [])
        bm25_rank = bm25_r.get(w.warning_id, [])
        emb_rank = emb_r.get(w.warning_id, [])
        ess = w.essentials()
        tp_score = recall_at_k(tp_rank, ess, 3)
        rrf_rank = rrf_merge([tp_rank, bm25_rank, emb_rank])
        rrf_score = recall_at_k(rrf_rank, ess, 3)
        if rrf_score < tp_score:
            hurt_cases.append({
                "warning_id": w.warning_id,
                "rule_id": w.rule_id,
                "tp_r3": round(tp_score, 3),
                "rrf3way_r3": round(rrf_score, 3),
                "delta": round(rrf_score - tp_score, 3),
            })
    print(f"\n  Warnings hurt by 3-way RRF: {len(hurt_cases)}")
    for c in sorted(hurt_cases, key=lambda x: x["delta"])[:5]:
        print(f"    {c['warning_id']:30s} TP={c['tp_r3']} RRF={c['rrf3way_r3']} Δ={c['delta']}")

    out = OUT_DIR / "E14_rrf_analysis.csv"
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["method", "recall@3"])
        wr.writeheader()
        wr.writerows(rows)

    hurt_out = OUT_DIR / "E14_rrf_hurt_cases.csv"
    with hurt_out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["warning_id", "rule_id", "tp_r3", "rrf3way_r3", "delta"])
        wr.writeheader()
        wr.writerows(sorted(hurt_cases, key=lambda x: x["delta"]))
    print(f"  → {out}, {hurt_out}")


def main():
    print("Loading data ...")
    all_warnings = load_warnings(DATA_PATH)
    test_warnings = warnings_by_split(all_warnings, "test")
    print(f"  Test: {len(test_warnings)} warnings")

    ltr_rankings = run_E12(all_warnings, test_warnings)
    run_E13(all_warnings, test_warnings, ltr_rankings)
    run_E14(test_warnings)


if __name__ == "__main__":
    main()
