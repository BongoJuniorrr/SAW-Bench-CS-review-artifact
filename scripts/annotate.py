"""Minimal interactive annotation CLI (paper §3.6).

Loads a JSONL of un-labeled warnings and walks the annotator through them in
randomized order, capturing the labels and rationales required by the public
schema. Output is one AnnotatorPass JSONL per annotator, suitable as input to
`scripts/build_dataset.py --annotations`.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import _bootstrap  # noqa: F401
from saw_bench_cs.construction.annotation import (
    AnnotatorPass,
    randomize_snippet_order,
    save_passes,
)
from saw_bench_cs.io import load_warnings


def _prompt_label(snippet) -> str:
    while True:
        ans = input(
            f"  [{snippet.snippet_id}] {snippet.type:<26s} "
            f"{snippet.path}:{snippet.line_start}-{snippet.line_end}\n"
            f"      {snippet.text[:120]}\n"
            f"      label (e=essential / h=helpful / Enter=irrelevant): "
        ).strip().lower()
        if ans in ("", "i", "irrelevant"):
            return "irrelevant"
        if ans in ("h", "helpful"):
            return "helpful"
        if ans in ("e", "essential"):
            return "essential"
        print("    please enter 'e', 'h', or empty for irrelevant")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--annotator", required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rng = random.Random(args.seed)
    warnings = load_warnings(args.data)
    rng.shuffle(warnings)

    passes: list[AnnotatorPass] = []
    for w in warnings:
        print(f"\n=== {w.warning_id} :: {w.rule_id} :: {w.warning_message[:120]}")
        ordered = randomize_snippet_order(w.candidate_snippets, args.seed)
        labels: dict[str, str] = {}
        rationales: dict[str, str] = {}
        essentials = 0
        for s in ordered:
            label = _prompt_label(s)
            if label == "essential":
                if essentials >= 3:
                    print("    cap of 3 essentials reached; downgrading to helpful")
                    label = "helpful"
                else:
                    essentials += 1
                    rationales[s.snippet_id] = input("      rationale: ").strip()
            labels[s.snippet_id] = label
        passes.append(AnnotatorPass(
            warning_id=w.warning_id,
            annotator=args.annotator,
            labels=labels,
            rationales=rationales,
        ))

    save_passes(passes, args.out)
    print(f"wrote {len(passes)} passes to {args.out}")


if __name__ == "__main__":
    main()
