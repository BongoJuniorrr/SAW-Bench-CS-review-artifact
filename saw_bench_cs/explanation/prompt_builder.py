"""Prompt construction for the three RQ4 strategies (paper §5.7)."""

from __future__ import annotations

from typing import Literal

from ..schema import LOCAL_SNIPPET_TYPES, CandidateSnippet, Warning


PromptStrategy = Literal["local_only", "full", "selected"]


_INSTRUCTIONS = (
    "Explain the SpotBugs warning below in 2-4 sentences. "
    "Reference the supplied evidence by snippet id, do not invent code, and "
    "be concise. If evidence is incomplete, say so explicitly."
)


def _format_snippets(snippets: list[CandidateSnippet]) -> str:
    parts = []
    for s in snippets:
        parts.append(
            f"[{s.snippet_id}] {s.type} @ {s.path}:{s.line_start}-{s.line_end}\n"
            f"{s.text.rstrip()}"
        )
    return "\n\n".join(parts)


def select_snippets(
    warning: Warning,
    strategy: PromptStrategy,
    selected_ids: list[str] | None = None,
    top_k: int = 3,
) -> list[CandidateSnippet]:
    if strategy == "local_only":
        return [
            s for s in warning.candidate_snippets
            if s.type in LOCAL_SNIPPET_TYPES
        ]
    if strategy == "full":
        return list(warning.candidate_snippets)
    if strategy == "selected":
        ids = (selected_ids or [])[:top_k]
        by_id = {s.snippet_id: s for s in warning.candidate_snippets}
        return [by_id[i] for i in ids if i in by_id]
    raise ValueError(f"unknown strategy {strategy!r}")


def build_prompt(
    warning: Warning,
    strategy: PromptStrategy,
    *,
    selected_ids: list[str] | None = None,
    top_k: int = 3,
) -> tuple[str, list[CandidateSnippet]]:
    """Return the prompt text and the snippets actually included."""
    snippets = select_snippets(warning, strategy, selected_ids=selected_ids, top_k=top_k)
    body = (
        f"Warning: {warning.warning_message}\n"
        f"Rule: {warning.rule_id} ({warning.category})\n"
        f"Location: {warning.file}:{warning.line}\n"
        f"Warning line: {warning.warning_context.warning_line}\n\n"
        f"Evidence:\n{_format_snippets(snippets)}\n\n"
        f"{_INSTRUCTIONS}"
    )
    return body, snippets
