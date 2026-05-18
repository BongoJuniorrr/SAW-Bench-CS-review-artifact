"""E11: LTR Feature Ablation — Q4 from paperreview.ai reviewer.

Ablates feature groups from the pairwise logistic LTR to quantify each group's contribution.
Output: results_extra/E11_ltr_ablation.csv
"""
from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from saw_bench_cs.io import load_warnings, warnings_by_split
from saw_bench_cs.evaluation.metrics import recall_at_k, ndcg_at_k, non_local_recovery_at_k
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
        print("  [embedding] sentence-transformers loaded")
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
        print(f"  [embedding] fallback: {e}")
        result = {}
        for w in warnings:
            result[w.warning_id] = {s.snippet_id: 0.0 for s in w.candidate_snippets}
        return result

def compute_full_features(warning, snippet, bm25_score, emb_score):
    """16-dim feature vector: type_onehot(11) + bm25 + emb + dist_norm + log_tok + is_local."""
    type_vec = [1 if snippet.type == t else 0 for t in SNIPPET_TYPES]
    mid = (snippet.line_start + snippet.line_end) / 2
    distance = abs(mid - warning.line) if warning.line > 0 else 0
    dist_norm = min(distance / 100.0, 10.0)
    log_tok = math.log(1 + snippet.token_count)
    is_local = 1 if snippet.type in LOCAL_SNIPPET_TYPES else 0
    return type_vec + [bm25_score, emb_score, dist_norm, log_tok, is_local]

# Feature index ranges (0-indexed):
# 0-10: type one-hot (11 dims for each SNIPPET_TYPE)
# 11: bm25_score
# 12: emb_cosine
# 13: dist_norm
# 14: log_tokens
# 15: is_local

def mask_features(feat_vec, mask):
    """Zero out dimensions where mask[i] == False."""
    return [v if mask[i] else 0.0 for i, v in enumerate(feat_vec)]

N_TYPES = len(SNIPPET_TYPES)  # 11

ABLATION_MASKS = {
    "full": [True] * (N_TYPES + 5),
    "no_type_onehot": [False] * N_TYPES + [True, True, True, True, True],
    "no_text": [True] * N_TYPES + [False, False, True, True, True],
    "no_proximity": [True] * N_TYPES + [True, True, False, True, True],
    "no_log_tokens": [True] * N_TYPES + [True, True, True, False, True],
    "no_is_local": [True] * N_TYPES + [True, True, True, True, False],
    "type_only": [True] * N_TYPES + [False, False, False, False, False],
    "text_only": [False] * N_TYPES + [True, True, False, False, False],
    "is_local_only": [False] * N_TYPES + [False, False, False, False, True],
}

def _eval_metrics(test_warnings, rankings_dict):
    r3, ndcg, nlr = [], [], []
    for w in test_warnings:
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

def run_ltr_with_mask(mask_name, mask, train_warnings, test_warnings,
                       train_bm25, test_bm25, train_emb, test_emb):
    rel_map = {"essential": 2, "helpful": 1, "irrelevant": 0}
    X_train, y_train = [], []
    for w in train_warnings:
        labels_map = {lbl.snippet_id: lbl.relevance for lbl in w.labels}
        bm25_sc = train_bm25.get(w.warning_id, {})
        emb_sc = train_emb.get(w.warning_id, {})
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
                fi = mask_features(
                    compute_full_features(w, si, bm25_sc.get(si.snippet_id, 0.0), emb_sc.get(si.snippet_id, 0.0)),
                    mask
                )
                fj = mask_features(
                    compute_full_features(w, sj, bm25_sc.get(sj.snippet_id, 0.0), emb_sc.get(sj.snippet_id, 0.0)),
                    mask
                )
                diff = [a - b for a, b in zip(fi, fj)]
                X_train.append(diff)
                y_train.append(1 if rel_i > rel_j else 0)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    X = np.array(X_train, dtype=float)
    y = np.array(y_train, dtype=int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)
    clf.fit(X_scaled, y)
    w_vec = clf.coef_[0]

    rankings = {}
    for w in test_warnings:
        bm25_sc = test_bm25.get(w.warning_id, {})
        emb_sc = test_emb.get(w.warning_id, {})
        scores = []
        for s in w.candidate_snippets:
            feat = mask_features(
                compute_full_features(w, s, bm25_sc.get(s.snippet_id, 0.0), emb_sc.get(s.snippet_id, 0.0)),
                mask
            )
            feat_scaled = scaler.transform([feat])[0]
            score = float(np.dot(w_vec, feat_scaled))
            scores.append((s.snippet_id, score))
        scores.sort(key=lambda x: -x[1])
        rankings[w.warning_id] = [sid for sid, _ in scores]

    m = _eval_metrics(test_warnings, rankings)
    m["ablation"] = mask_name
    print(f"  [{mask_name:20s}]  R@3={m['recall@3']:.4f}  nDCG@5={m['ndcg@5']:.4f}  NLR@3={m['non_local_recovery@3']:.4f}")
    return m


def main():
    print("Loading data ...")
    all_warnings = load_warnings(DATA_PATH)
    train_warnings = warnings_by_split(all_warnings, "train")
    test_warnings = warnings_by_split(all_warnings, "test")
    print(f"  Train: {len(train_warnings)} warnings | Test: {len(test_warnings)} warnings")

    print("Computing BM25 scores ...")
    train_bm25 = {w.warning_id: bm25_scores_for_warning(w) for w in train_warnings}
    test_bm25 = {w.warning_id: bm25_scores_for_warning(w) for w in test_warnings}

    print("Computing embedding scores (train) ...")
    train_emb = _embed_all(train_warnings)
    print("Computing embedding scores (test) ...")
    test_emb = _embed_all(test_warnings)

    results = []
    for mask_name, mask in ABLATION_MASKS.items():
        print(f"\nRunning ablation: {mask_name}")
        m = run_ltr_with_mask(
            mask_name, mask, train_warnings, test_warnings,
            train_bm25, test_bm25, train_emb, test_emb
        )
        results.append(m)

    out = OUT_DIR / "E11_ltr_ablation.csv"
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["ablation", "recall@3", "ndcg@5", "non_local_recovery@3"])
        wr.writeheader()
        wr.writerows(results)
    print(f"\n→ Saved: {out}")
    for r in results:
        print(f"  {r['ablation']:20s}  R@3={r['recall@3']:.4f}  NLR@3={r['non_local_recovery@3']:.4f}")


if __name__ == "__main__":
    main()
