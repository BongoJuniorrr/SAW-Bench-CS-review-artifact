# SAW-Bench-CS Annotation Rubric

This rubric defines the relevance labels used by the artifact-labeling passes and
can also be used by future human annotators.

## Setup shown for every warning

- The SpotBugs warning message and the warning line.
- The enclosing method.
- 5–10 candidate snippets in randomized order, each with its type and source path.

## Labels

For every snippet pick exactly one of:

- **essential** — needed to understand why the analyzer raised this warning, or to judge whether the warning is plausible. If you remove this snippet, an explanation of the warning would be incomplete or misleading.
- **helpful** — useful background that makes the explanation easier to follow but is not strictly required.
- **irrelevant** — redundant with another snippet, off-topic, or distracting.

## Rules

- At most 3 snippets per warning may be marked **essential**.
- Every **essential** label must include a one-sentence rationale describing the role of the snippet.
- Helpful labels may be unbounded.
- Default is **irrelevant**.
- Do **not** decide whether the warning identifies a real defect. Only judge whether the snippet helps explain it.

## Agreement audit

- Agreement is recomputed from `annotation/annotator_passes.jsonl`.
- Disagreements are retained via `annotator_count` and never adjudicated by an expert.

## Common pitfalls

- Marking the warning line "essential" by reflex even when the enclosing method already makes the cause obvious.
- Marking similar-code or history snippets "essential" because they are related; reserve essential for evidence that explains the warning.
- Forgetting the rationale on essential labels — the validator will reject these.
