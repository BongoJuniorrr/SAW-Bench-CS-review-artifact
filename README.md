# SAW-Bench-CS

Source-code-only reference implementation for **SAW-Bench-CS: A Context Selection Artifact for SpotBugs Warning Explanation in Java Utility Libraries** (ESEM 2026 submission).

The paper reports an artifact with 403 SpotBugs warnings from 33 open-source Java utility and infrastructure libraries paired with 3,099 candidate evidence snippets and snippet-level relevance labels. This repository ships the source code, configuration, labeled benchmark JSONL, raw annotation-pass JSONL, canonical reproduced result files, and diagnostic outputs needed to validate and evaluate the artifact.

- A reproducible six-stage dataset construction pipeline.
- A typed JSONL schema and consistency validator.
- Four lightweight retrieval baselines, a random-permutation floor, and an optional endpoint LLM reranker path.
- Metrics from paper §5: Recall@k, MRR, nDCG@5, plus the four diagnostic metrics in Table 3.
- A reproducible RQ4 explanation-quality protocol with generated explanations, deterministic rubric scores, and an endpoint-compatible rerun script.
- Released A/B pass files, a merge script, and an agreement comparison script for reproducing the benchmark labels.

Canonical paper outputs are stored under `results_reproduced/`. Extra diagnostic tables used by the supplement are stored under `results_extra/`. The quickstart commands below write to `results_local/` so reruns do not overwrite the shipped reference outputs.

## Quickstart

```bash
pip install -r requirements.txt

# Recreate the tracked labeled artifact by merging the released A/B pass files.
python scripts/create_labeled_artifact_from_passes.py \
  --left-pass annotation/artifact_labeler_A_complete_pass.jsonl \
  --right-pass annotation/artifact_labeler_B_pass.jsonl \
  --out data/saw_bench_cs.jsonl \
  --passes-out annotation/annotator_passes.jsonl

# Recompute annotation agreement from raw per-pass labels.
python scripts/compute_annotation_agreement.py \
  --data data/saw_bench_cs.jsonl \
  --passes annotation/annotator_passes.jsonl

# Compare the released A/B pass files.
python scripts/compare_annotation_passes.py \
  --data data/saw_bench_cs.jsonl \
  --left annotation/artifact_labeler_A_complete_pass.jsonl \
  --right annotation/artifact_labeler_B_pass.jsonl \
  --left-name artifact_labeler_A \
  --right-name artifact_labeler_B \
  --out-json annotation/manual_audit_agreement.json \
  --out-csv annotation/manual_audit_labels.csv

# Validate against the public schema.
python scripts/validate_dataset.py data/saw_bench_cs.jsonl

# Reproduce test-split baselines.
# The LLM reranker is included only when SAW_LLM_ENDPOINT is configured.
python scripts/run_baselines.py \
  --data data/saw_bench_cs.jsonl \
  --prior-split validation \
  --split test \
  --out results_local/

# Random-permutation artifact floor for Table 4.
python scripts/run_random_baseline.py \
  --data data/saw_bench_cs.jsonl \
  --split test \
  --out results_local/random_baseline.json

# Diagnostic metrics (Table 3).
python scripts/run_diagnostics.py \
  --data data/saw_bench_cs.jsonl \
  --rankings results_local/rankings/ \
  --out results_local/diagnostics.csv

# RQ4 explanation-quality study.
.venv/bin/python scripts/generate_rq4_artifacts.py \
  --data data/saw_bench_cs.jsonl \
  --split test \
  --explanations-out results_local/explanations.csv \
  --scored-out results_local/explanations_scored.csv \
  --calibration-out annotation/rq4_calibration_log.csv \
  --summary-out results_local/explanations_summary.json

# Optional endpoint-LLM generation.
SAW_LLM_ENDPOINT=http://localhost:11434/v1/chat/completions \
  python scripts/run_explanation_study.py \
  --data data/saw_bench_cs.jsonl \
  --n 100 \
  --out results_local/explanations.csv

# Or aggregate your own hand-scored CSV without calling an LLM.
python scripts/run_explanation_study.py --scores path/to/explanations_scored.csv
```

## Reproducibility

- All stages are deterministic given the random seed in `configs/eval.yaml` (default 7).
- Pinned commits for the 105-project public Java execution set live in `configs/projects_public_java_100.yaml`; `configs/projects.yaml` is a compatibility alias for the same v23 configuration.
- Project-disjoint splits live in `configs/splits_public_java_100.yaml`; `configs/splits.yaml` is the matching compatibility alias.
- The validator enforces every constraint stated in paper §3.6 (annotation consistency).
- The construction pipeline now compiles each Maven/Gradle checkout before invoking SpotBugs; use `--skip-build` only when the checkouts are already compiled.
- The source repository includes the labeled JSONL, raw A/B annotation-pass JSONL, baseline metrics, rankings, random-permutation summary, RQ4 deterministic explanation-quality artifacts, supplementary diagnostic outputs, and runtime provenance needed to reproduce the core tables. Endpoint-LLM RQ4 generation still requires a configured LLM endpoint or a hand-scored CSV.

## License

CC BY 4.0 for the dataset; MIT for the code under `saw_bench_cs/`.
