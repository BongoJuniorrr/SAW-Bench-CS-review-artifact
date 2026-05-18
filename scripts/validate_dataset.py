"""Run consistency checks against a SAW-Bench-CS JSONL file."""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from saw_bench_cs.construction.consistency_checks import validate_dataset
from saw_bench_cs.io import load_warnings


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--allow-unlabeled", action="store_true",
                   help="Validate source-derived records before annotation labels exist.")
    args = p.parse_args()

    warnings = load_warnings(args.path)
    report = validate_dataset(warnings, require_labels=not args.allow_unlabeled)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
