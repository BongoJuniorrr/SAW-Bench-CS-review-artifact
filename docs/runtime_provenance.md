# Runtime Measurement Provenance

This file documents the local engineering measurements summarized in the paper's
cost-budget table. These timings are intended to bound the practical cost of
using the artifact; they are not statistical performance claims.

## Measurement Environment

- Date recorded: 2026-04-28/2026-04-29
- Host class: single workstation
- CPU: 16-core AMD Ryzen 9 5950X
- Memory: 64 GB RAM
- GPU: NVIDIA RTX 3090 24 GB
- Python: 3.11
- Java: OpenJDK 17
- Build tool: Maven
- Static analyzer: SpotBugs
- Non-LLM stages: single process

## Reported Local Timings

| Stage | Local wall-clock cost | Notes |
| --- | ---: | --- |
| SpotBugs analysis, 105-project public Java configuration with 33 projects producing retained warnings | See `docs/public_projects.md` | Includes analyzer execution over compiled project checkouts. Project-level misses are summarized in `docs/project_errors_public_java_100.txt`. |
| Candidate generation, full 403-warning artifact | See `docs/public_projects.md` | End-to-end extraction over the successful project set. |
| Candidate generation, per-warning p95 | 11 s | Computed from the local extraction run. |
| BM25 ranking, held-out test split | <1 s | Reproduced by `scripts/run_baselines.py`. |
| Embedding ranking with all-MiniLM-L6-v2, held-out test split | See shipped `results_reproduced/metrics.csv` | Uses the local Sentence-Transformers model path reported in the paper. If Sentence-Transformers is unavailable, the code falls back to a deterministic hashed embedding and labels that output separately. |
| Type-priority ranking, held-out test split | 1 s | Reproduced by `scripts/run_baselines.py`. |
| End-to-end default test-split baseline evaluation | <1 s | Excludes the endpoint LLM reranker unless `SAW_LLM_ENDPOINT` is configured. |

## Reproduction Commands

The 105-project public Java run record and project-level outcomes are
documented in `docs/public_projects.md`. The commands below reproduce the
shipped labeled artifact and default evaluation tables from the tracked data.

```bash
python scripts/create_labeled_artifact_from_passes.py \
  --left-pass annotation/artifact_labeler_A_complete_pass.jsonl \
  --right-pass annotation/artifact_labeler_B_pass.jsonl \
  --out data/saw_bench_cs.jsonl \
  --passes-out annotation/annotator_passes.jsonl

python scripts/compute_annotation_agreement.py \
  --data data/saw_bench_cs.jsonl \
  --passes annotation/annotator_passes.jsonl

python scripts/validate_dataset.py data/saw_bench_cs.jsonl

python scripts/run_baselines.py \
  --data data/saw_bench_cs.jsonl \
  --prior-split validation \
  --split test \
  --out results_local/
```

The exact construction timings depend on hardware, local Maven dependency
caches, and SpotBugs execution behavior. For that reason the paper reports them
as local cost-budget measurements rather than as inferential results.
