"""Public JSONL schema and Python dataclasses for SAW-Bench-CS records.

Mirrors the schema described in paper §3.5 and the constraints enforced
by saw_bench_cs.construction.consistency_checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable, Literal, Optional


SnippetType = Literal[
    "warning_line",
    "enclosing_method",
    "enclosing_class",
    "caller",
    "callee",
    "field_or_type_declaration",
    "annotation_or_contract",
    "test",
    "configuration",
    "similar_code",
    "history_or_diff",
]

SNIPPET_TYPES: tuple[SnippetType, ...] = (
    "warning_line",
    "enclosing_method",
    "enclosing_class",
    "caller",
    "callee",
    "field_or_type_declaration",
    "annotation_or_contract",
    "test",
    "configuration",
    "similar_code",
    "history_or_diff",
)

# Local types as defined in paper §5.3 (non-local recovery diagnostic):
# "Non-local snippets exclude warning_line and enclosing_method."
LOCAL_SNIPPET_TYPES: frozenset[SnippetType] = frozenset(
    {"warning_line", "enclosing_method"}
)

Relevance = Literal["essential", "helpful", "irrelevant"]
Split = Literal["train", "validation", "test"]
Severity = Literal["high", "medium", "low"]

# Graded relevance gain used by nDCG and token-normalized utility (paper §5.3).
RELEVANCE_GAIN: dict[Relevance, int] = {
    "essential": 2,
    "helpful": 1,
    "irrelevant": 0,
}

MIN_SNIPPETS_PER_WARNING = 5
MAX_SNIPPETS_PER_WARNING = 10
MAX_ESSENTIAL_PER_WARNING = 3


@dataclass
class WarningContext:
    warning_line: str
    complete_statement: str
    method_excerpt: str


@dataclass
class CandidateSnippet:
    snippet_id: str
    type: SnippetType
    path: str
    line_start: int
    line_end: int
    text: str
    token_count: int
    generation_method: str


@dataclass
class RelevanceLabel:
    snippet_id: str
    relevance: Relevance
    annotator_count: int  # number of labeling passes that agreed (1 or 2)
    rationale: Optional[str] = None  # required iff relevance == "essential"


@dataclass
class Warning:
    """One JSONL record. Field order matches paper §3.5."""

    warning_id: str
    project: str
    commit: str
    tool: str
    rule_id: str
    category: str
    severity: Severity
    file: str
    line: int
    warning_message: str
    warning_context: WarningContext
    candidate_snippets: list[CandidateSnippet]
    labels: list[RelevanceLabel]
    split: Split

    # ------------------------------------------------------------------ helpers
    def snippet(self, snippet_id: str) -> CandidateSnippet:
        for snippet in self.candidate_snippets:
            if snippet.snippet_id == snippet_id:
                return snippet
        raise KeyError(snippet_id)

    def label_of(self, snippet_id: str) -> Relevance:
        for lbl in self.labels:
            if lbl.snippet_id == snippet_id:
                return lbl.relevance
        return "irrelevant"

    def essentials(self) -> set[str]:
        return {lbl.snippet_id for lbl in self.labels if lbl.relevance == "essential"}

    def helpfuls(self) -> set[str]:
        return {lbl.snippet_id for lbl in self.labels if lbl.relevance == "helpful"}

    def useful(self) -> set[str]:
        return self.essentials() | self.helpfuls()

    def to_dict(self) -> dict:
        return asdict(self)

    # ----------------------------------------------------------------- factory
    @classmethod
    def from_dict(cls, raw: dict) -> "Warning":
        ctx = raw["warning_context"]
        return cls(
            warning_id=raw["warning_id"],
            project=raw["project"],
            commit=raw["commit"],
            tool=raw["tool"],
            rule_id=raw["rule_id"],
            category=raw["category"],
            severity=raw["severity"],
            file=raw["file"],
            line=int(raw["line"]),
            warning_message=raw["warning_message"],
            warning_context=WarningContext(
                warning_line=ctx["warning_line"],
                complete_statement=ctx["complete_statement"],
                method_excerpt=ctx["method_excerpt"],
            ),
            candidate_snippets=[
                CandidateSnippet(**s) for s in raw["candidate_snippets"]
            ],
            labels=[RelevanceLabel(**lbl) for lbl in raw["labels"]],
            split=raw["split"],
        )


# ------------------------- JSON Schema (informational, used by validator) ----

JSON_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "SAW-Bench-CS Warning Record",
    "type": "object",
    "required": [
        "warning_id", "project", "commit", "tool", "rule_id", "category",
        "severity", "file", "line", "warning_message", "warning_context",
        "candidate_snippets", "labels", "split",
    ],
    "properties": {
        "warning_id": {"type": "string", "minLength": 1},
        "project": {"type": "string", "minLength": 1},
        "commit": {"type": "string", "minLength": 1},
        "tool": {"type": "string", "minLength": 1},
        "rule_id": {"type": "string", "minLength": 1},
        "category": {"type": "string", "minLength": 1},
        "severity": {"enum": ["high", "medium", "low"]},
        "file": {"type": "string", "minLength": 1},
        "line": {"type": "integer", "minimum": 1},
        "warning_message": {"type": "string", "minLength": 1},
        "warning_context": {
            "type": "object",
            "required": ["warning_line", "complete_statement", "method_excerpt"],
            "properties": {
                "warning_line": {"type": "string"},
                "complete_statement": {"type": "string"},
                "method_excerpt": {"type": "string"},
            },
        },
        "candidate_snippets": {
            "type": "array",
            "minItems": MIN_SNIPPETS_PER_WARNING,
            "maxItems": MAX_SNIPPETS_PER_WARNING,
            "items": {
                "type": "object",
                "required": [
                    "snippet_id", "type", "path", "line_start", "line_end",
                    "text", "token_count", "generation_method",
                ],
                "properties": {
                    "snippet_id": {"type": "string"},
                    "type": {"enum": list(SNIPPET_TYPES)},
                    "path": {"type": "string"},
                    "line_start": {"type": "integer", "minimum": 1},
                    "line_end": {"type": "integer", "minimum": 1},
                    "text": {"type": "string"},
                    "token_count": {"type": "integer", "minimum": 0},
                    "generation_method": {"type": "string"},
                },
            },
        },
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["snippet_id", "relevance", "annotator_count"],
                "properties": {
                    "snippet_id": {"type": "string"},
                    "relevance": {"enum": ["essential", "helpful", "irrelevant"]},
                    "annotator_count": {"type": "integer", "minimum": 1, "maximum": 2},
                    "rationale": {"type": ["string", "null"]},
                },
            },
        },
        "split": {"enum": ["train", "validation", "test"]},
    },
}


def all_snippet_ids(warning: Warning) -> Iterable[str]:
    return (s.snippet_id for s in warning.candidate_snippets)
