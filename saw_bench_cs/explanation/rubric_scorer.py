"""Lightweight 4-criterion rubric scorer (paper §5.7).

Each explanation is scored 1-5 by aggregating four binary criteria into a
1-5 scale (1 = none satisfied, 5 = all four satisfied + cohesive). The
unsupported-claim rate is reported separately.

The scorer can run locally with simple heuristics for offline evaluation, or
delegate to an LLM-as-judge through `LLMClient`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from ..schema import CandidateSnippet
from .llm_client import LLMClient


@dataclass
class RubricScores:
    quality: int            # 1-5
    mentions_cause: bool
    references_evidence: bool
    no_unsupported_claims: bool
    concise: bool

    def to_dict(self) -> dict:
        return {
            "quality": self.quality,
            "mentions_cause": self.mentions_cause,
            "references_evidence": self.references_evidence,
            "no_unsupported_claims": self.no_unsupported_claims,
            "concise": self.concise,
        }


_HEURISTIC_PROMPT = """\
You are scoring a static-analysis warning explanation. Reply with strict JSON:

{{"mentions_cause": bool, "references_evidence": bool, "no_unsupported_claims": bool, "concise": bool}}

Criteria:
  mentions_cause      : the explanation states why the analyzer raised the warning
  references_evidence : it cites at least one supplied snippet id
  no_unsupported_claims: it does not assert facts beyond the supplied evidence
  concise              : at most 4 sentences

Warning rule: {rule_id}
Warning text: {warning_message}
Snippets supplied: {snippet_ids}
Explanation: {explanation}
"""


def _heuristic_scores(
    explanation: str,
    snippets: Iterable[CandidateSnippet],
    warning_message: str,
) -> RubricScores:
    snippet_ids = [s.snippet_id for s in snippets]
    text = (explanation or "").strip()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    references_evidence = any(sid.lower() in text.lower() for sid in snippet_ids)
    concise = 0 < len(sentences) <= 4
    cause_words = re.findall(r"\b\w+\b", warning_message.lower())[:6]
    mentions_cause = any(w in text.lower() for w in cause_words if len(w) > 3)
    # Crude unsupported-claim heuristic: if the explanation mentions identifiers
    # that aren't in any supplied snippet, flag it.
    snippet_blob = " ".join(s.text for s in snippets)
    explanation_idents = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", text))
    snippet_idents = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", snippet_blob))
    unsupported = explanation_idents - snippet_idents - set(cause_words)
    no_unsupported_claims = len(unsupported) <= 4

    score = 1 + sum([mentions_cause, references_evidence, no_unsupported_claims, concise])
    return RubricScores(
        quality=score,
        mentions_cause=mentions_cause,
        references_evidence=references_evidence,
        no_unsupported_claims=no_unsupported_claims,
        concise=concise,
    )


def _llm_scores(
    explanation: str,
    snippets: Iterable[CandidateSnippet],
    warning_message: str,
    rule_id: str,
    client: LLMClient,
) -> Optional[RubricScores]:
    snippet_ids = [s.snippet_id for s in snippets]
    prompt = _HEURISTIC_PROMPT.format(
        rule_id=rule_id,
        warning_message=warning_message,
        snippet_ids=snippet_ids,
        explanation=explanation,
    )
    raw = client.complete(prompt)
    if not raw:
        return None
    match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except ValueError:
        return None
    flags = {k: bool(payload.get(k, False)) for k in
             ("mentions_cause", "references_evidence", "no_unsupported_claims", "concise")}
    return RubricScores(
        quality=1 + sum(flags.values()),
        **flags,
    )


def score_explanation(
    explanation: str,
    snippets: Iterable[CandidateSnippet],
    warning_message: str,
    *,
    rule_id: str = "",
    client: Optional[LLMClient] = None,
) -> RubricScores:
    """Score an explanation. Uses LLM-as-judge if `client` is configured."""
    if client is not None and client.is_configured():
        scores = _llm_scores(explanation, snippets, warning_message, rule_id, client)
        if scores is not None:
            return scores
    return _heuristic_scores(explanation, snippets, warning_message)


def aggregate_scores(scores: list[RubricScores]) -> dict[str, float]:
    if not scores:
        return {"quality": 0.0, "unsupported_claim_rate": 0.0}
    n = len(scores)
    avg_quality = sum(s.quality for s in scores) / n
    unsupported_rate = sum(1 for s in scores if not s.no_unsupported_claims) / n
    return {
        "quality": avg_quality,
        "unsupported_claim_rate": unsupported_rate,
    }
