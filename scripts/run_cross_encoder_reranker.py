"""Cross-encoder reranker baseline for SAW-Bench-CS.

Ranks BM25 top-k candidates using a sentence-transformers CrossEncoder and
evaluates the same metric set as the other retrieval baselines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from saw_bench_cs.evaluation.metrics import aggregate
from saw_bench_cs.io import load_warnings, warnings_by_split


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu", first_stage: str = "bm25", top_k_candidates: int = 7):
        self.name = f"cross_encoder_{model_name.split('/')[-1]}"
        self.model_name = model_name
        self.device = device
        self.first_stage = first_stage
        self.top_k_candidates = top_k_candidates
        self._model = None

    def fit(self, warnings):
        # No training; loading is deferred until first rank call.
        return None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rank(self, warning):
        from saw_bench_cs.baselines import BM25Ranker, LocalFirstRanker

        if self.first_stage == "local_first":
            stage = LocalFirstRanker().rank(warning)
        else:
            stage = BM25Ranker().rank(warning)
        if not stage:
            return []
        head, tail = stage[: self.top_k_candidates], stage[self.top_k_candidates :]
        snippets = {s.snippet_id: s for s in warning.candidate_snippets}
        pairs = [
            (warning.warning_message + " " + warning.rule_id + " " + warning.category, snippets[sid].text)
            for sid in head
            if sid in snippets
        ]
        model = self._ensure_model()
        scores = model.predict(pairs)
        order = sorted(range(len(head)), key=lambda i: (-float(scores[i]), i))
        new_head = [head[i] for i in order]
        return new_head + tail


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/saw_bench_cs.jsonl")
    p.add_argument("--split", default="test")
    p.add_argument("--first-stage", choices=["bm25", "local_first"], default="bm25")
    p.add_argument("--top-k-candidates", type=int, default=7)
    p.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results_gpu/cross_encoder_reranker")
    args = p.parse_args()

    warnings = warnings_by_split(load_warnings(args.data), args.split)
    ranker = CrossEncoderReranker(
        model_name=args.model,
        device=args.device,
        first_stage=args.first_stage,
        top_k_candidates=args.top_k_candidates,
    )
    ranker.fit(warnings)
    rankings = {w.warning_id: ranker.rank(w) for w in warnings}
    metrics = aggregate(warnings, rankings)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir.with_suffix(".csv") if out_dir.suffix else out_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as fh:
        import csv

        keys = list(metrics.keys())
        writer = csv.writer(fh)
        writer.writerow(["method", *keys])
        writer.writerow([ranker.name, *(metrics[k] for k in keys)])

    rankings_dir = out_dir / "rankings"
    rankings_dir.mkdir(parents=True, exist_ok=True)
    with (rankings_dir / f"{ranker.name}.jsonl").open("w", encoding="utf-8") as fh:
        import json

        for warning_id, ranking in rankings.items():
            fh.write(json.dumps({"warning_id": warning_id, "ranking": ranking}))
            fh.write("\n")

    print(f"wrote {metrics_path}")
    print(f"wrote {rankings_dir / f'{ranker.name}.jsonl'}")


if __name__ == "__main__":
    main()
