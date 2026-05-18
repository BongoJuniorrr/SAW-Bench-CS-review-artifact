"""Create a labeled benchmark by merging two explicit annotation pass files."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from saw_bench_cs.construction.annotation import (
    enforce_essential_cap,
    load_passes,
    merge_passes,
    save_passes,
)
from saw_bench_cs.io import load_warnings, save_warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/saw_bench_cs.jsonl")
    parser.add_argument("--left-pass", required=True)
    parser.add_argument("--right-pass", required=True)
    parser.add_argument("--out", default="data/saw_bench_cs.jsonl")
    parser.add_argument("--passes-out", default="annotation/annotator_passes.jsonl")
    args = parser.parse_args()

    warnings = load_warnings(args.input)
    passes = load_passes(args.left_pass) + load_passes(args.right_pass)
    by_warning = {}
    for annotation_pass in passes:
        by_warning.setdefault(annotation_pass.warning_id, []).append(annotation_pass)

    for warning in warnings:
        pair = by_warning.get(warning.warning_id, [])
        if len(pair) != 2:
            raise SystemExit(f"{warning.warning_id}: expected exactly two annotation passes")
        warning.labels = enforce_essential_cap(merge_passes(pair))

    save_warnings(warnings, args.out)
    save_passes(passes, args.passes_out)
    print(f"wrote {Path(args.out)}")
    print(f"wrote {Path(args.passes_out)}")


if __name__ == "__main__":
    main()
