"""Run the six-stage construction pipeline and emit the JSONL benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from saw_bench_cs.construction.pipeline import run_pipeline


def main() -> None:
    p = argparse.ArgumentParser(description="Build SAW-Bench-CS from source projects.")
    p.add_argument("--config", default="configs/projects.yaml")
    p.add_argument("--splits", default="configs/splits.yaml")
    p.add_argument("--work-root", default="data/build")
    p.add_argument("--annotations", default=None,
                   help="Optional path to annotation passes JSONL.")
    p.add_argument("--out", default="data/saw_bench_cs.jsonl")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--skip-build", action="store_true",
                   help="Assume checkouts are already compiled before SpotBugs runs.")
    p.add_argument("--allow-unlabeled", action="store_true",
                   help="Write source-derived records before annotation labels exist.")
    p.add_argument("--drop-short-candidates", action="store_true",
                   help="Drop warnings with fewer than five generated candidate snippets.")
    p.add_argument("--continue-on-project-error", action="store_true",
                   help="Skip projects that fail checkout, build, or SpotBugs.")
    args = p.parse_args()

    out = run_pipeline(
        config_path=args.config,
        splits_path=args.splits,
        work_root=args.work_root,
        out_path=args.out,
        annotation_passes_path=args.annotations,
        seed=args.seed,
        build_projects=not args.skip_build,
        require_labels=not args.allow_unlabeled,
        drop_short_candidates=args.drop_short_candidates,
        continue_on_project_error=args.continue_on_project_error,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
