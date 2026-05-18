"""Create a transparent artifact_labeler_B audit pass.

This pass is deliberately not described as human annotation. It applies an
independent, rule-aware labeling policy over the released candidate snippets and
writes one JSONL record per warning for audit comparison.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401
from saw_bench_cs.io import load_warnings
from saw_bench_cs.schema import CandidateSnippet, Warning


LOCAL_TYPES = {"warning_line", "enclosing_method"}
DECL_TYPES = {"field_or_type_declaration", "annotation_or_contract", "enclosing_class"}
CALL_TYPES = {"caller", "callee"}
REPO_TYPES = {"test", "similar_code", "configuration", "history_or_diff"}


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def rule_family(warning: Warning) -> set[str]:
    rule = warning.rule_id.upper()
    msg = warning.warning_message.lower()
    fam: set[str] = set()

    if any(x in rule for x in ("EI", "MS_", "UWF_", "URF_", "SE_", "SIC_", "HE_", "EQ_", "CN_", "CT_", "CO_")):
        fam.add("declaration")
    if any(x in msg for x in ("mutable", "field", "serial", "singleton", "subclass", "clone", "equals", "hashcode", "class ")):
        fam.add("declaration")

    if any(x in rule for x in ("THROWS", "NP_", "RCN_", "BC_", "DLS_", "RV_")):
        fam.add("contract")
    if any(x in msg for x in ("nonnull", "nullable", "contract", "override", "throws", "return value", "null")):
        fam.add("contract")

    if any(x in rule for x in ("DM_", "DMI_", "RV_", "OBL_", "OS_", "SQL_", "NP_")):
        fam.add("call")
    if any(x in msg for x in ("called", "method", "stream", "resource", "close", "database", "return value")):
        fam.add("call")

    if any(x in rule for x in ("NP_", "EQ_", "HE_", "RCN_", "BC_")):
        fam.add("usage")
    return fam


def interesting_warning_line(snippet: CandidateSnippet) -> bool:
    text = clean(snippet.text)
    if not text:
        return False
    # A bare annotation, brace, import, or signature is usually not sufficient
    # by itself to explain the warning.
    if text in {"{", "}", ");"}:
        return False
    if text.startswith("@") and len(text) < 80:
        return False
    if re.match(r"^(public|private|protected)?\s*(static\s+)?(final\s+)?class\b", text):
        return False
    return any(tok in text for tok in ("=", "return", "throw", "new ", "==", "!=", ".", "(", "null", "synchronized"))


def method_has_context(snippet: CandidateSnippet, warning_line: str) -> bool:
    text = clean(snippet.text)
    line = clean(warning_line)
    if len(text) < max(80, len(line) + 35):
        return False
    return text.count(";") > 1 or any(x in text for x in ("if ", "for ", "while ", "try ", "catch ", "return"))


def token_overlap_score(warning: Warning, snippet: CandidateSnippet) -> int:
    haystack = clean(snippet.text).lower()
    words = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", warning.warning_message + " " + warning.rule_id))
    stop = {"warning", "method", "class", "field", "value", "return", "this", "that", "with", "from", "will"}
    words -= stop
    return sum(1 for word in words if word.lower() in haystack)


def proximity_score(warning: Warning, snippet: CandidateSnippet) -> int:
    mid = (snippet.line_start + snippet.line_end) / 2
    return -int(abs(mid - warning.line))


def rationale_for(warning: Warning, snippet: CandidateSnippet) -> str:
    stype = snippet.type.replace("_", " ")
    text = clean(snippet.text)
    if snippet.type == "warning_line":
        return (
            f"The reported line shows the concrete operation tied to {warning.rule_id}, "
            f"so it anchors the warning explanation at {Path(snippet.path).name}:{snippet.line_start}."
        )
    if snippet.type == "enclosing_method":
        return (
            f"The enclosing method adds the control/data context around the reported line, "
            f"which is needed to explain why {warning.rule_id} is plausible here."
        )
    if snippet.type == "field_or_type_declaration":
        return (
            f"The declaration snippet exposes the relevant type or field property that the "
            f"{warning.rule_id} warning depends on."
        )
    if snippet.type == "annotation_or_contract":
        return (
            f"The contract/annotation snippet states an API obligation that changes how "
            f"the {warning.rule_id} warning should be interpreted."
        )
    if snippet.type == "enclosing_class":
        return (
            f"The class excerpt gives class-level structure needed to interpret the "
            f"{warning.rule_id} warning beyond the local statement."
        )
    if snippet.type in CALL_TYPES:
        return (
            f"The {stype} snippet shows related call behavior that affects the explanation "
            f"of {warning.rule_id}."
        )
    return (
        f"The {stype} snippet supplies warning-specific context for {warning.rule_id}: "
        f"{text[:90]}"
    )


def choose_essentials(warning: Warning) -> list[CandidateSnippet]:
    snippets = warning.candidate_snippets
    by_type = {s.type: s for s in snippets}
    fam = rule_family(warning)
    warning_line_text = clean(by_type.get("warning_line", CandidateSnippet("", "warning_line", "", 1, 1, "", 0, "")).text)

    candidates: list[tuple[int, CandidateSnippet]] = []
    for snippet in snippets:
        score = token_overlap_score(warning, snippet)
        if snippet.type == "warning_line":
            score += 32 if interesting_warning_line(snippet) else 10
        elif snippet.type == "enclosing_method":
            score += 26 if method_has_context(snippet, warning_line_text) else 11
        elif snippet.type == "field_or_type_declaration":
            score += 30 if "declaration" in fam else 14
        elif snippet.type == "annotation_or_contract":
            score += 30 if "contract" in fam else 12
        elif snippet.type == "enclosing_class":
            score += 18 if "declaration" in fam else 7
        elif snippet.type in CALL_TYPES:
            score += 20 if "call" in fam else 5
        elif snippet.type in REPO_TYPES:
            score += 14 if "usage" in fam else 2
        score += max(-12, proximity_score(warning, snippet) // 25)
        candidates.append((score, snippet))

    ordered = [s for _, s in sorted(candidates, key=lambda item: (item[0], -item[1].token_count), reverse=True)]
    essentials: list[CandidateSnippet] = []
    used_types: set[str] = set()

    # Prefer at least one local anchor when it is informative, but do not force
    # both local snippets to be essential for every warning.
    for snippet in ordered:
        if snippet.type == "warning_line" and interesting_warning_line(snippet):
            essentials.append(snippet)
            used_types.add(snippet.type)
            break

    for snippet in ordered:
        if len(essentials) >= 3:
            break
        if snippet.snippet_id in {s.snippet_id for s in essentials}:
            continue
        if snippet.type == "enclosing_method" and method_has_context(snippet, warning_line_text):
            essentials.append(snippet)
            used_types.add(snippet.type)
            break

    for snippet in ordered:
        if len(essentials) >= 3:
            break
        if snippet.snippet_id in {s.snippet_id for s in essentials}:
            continue
        if snippet.type in DECL_TYPES and ("declaration" in fam or "contract" in fam):
            essentials.append(snippet)
            used_types.add(snippet.type)
            break

    for snippet in ordered:
        if len(essentials) >= 3:
            break
        if snippet.snippet_id in {s.snippet_id for s in essentials}:
            continue
        if snippet.type in CALL_TYPES and "call" in fam and token_overlap_score(warning, snippet) > 0:
            essentials.append(snippet)
            used_types.add(snippet.type)
            break

    if not essentials:
        essentials = ordered[:1]
    return essentials[:3]


def label_warning(warning: Warning) -> dict[str, object]:
    essentials = choose_essentials(warning)
    essential_ids = {s.snippet_id for s in essentials}
    fam = rule_family(warning)

    labels: dict[str, str] = {}
    rationales: dict[str, str] = {}
    for snippet in warning.candidate_snippets:
        if snippet.snippet_id in essential_ids:
            labels[snippet.snippet_id] = "essential"
            rationales[snippet.snippet_id] = rationale_for(warning, snippet)
            continue

        overlap = token_overlap_score(warning, snippet)
        if snippet.type in LOCAL_TYPES:
            labels[snippet.snippet_id] = "helpful"
        elif snippet.type in DECL_TYPES and ("declaration" in fam or "contract" in fam or overlap > 0):
            labels[snippet.snippet_id] = "helpful"
        elif snippet.type in CALL_TYPES and ("call" in fam or overlap > 1):
            labels[snippet.snippet_id] = "helpful"
        elif snippet.type in {"test", "similar_code"} and ("usage" in fam or overlap > 2):
            labels[snippet.snippet_id] = "helpful"
        else:
            labels[snippet.snippet_id] = "irrelevant"

    return {
        "warning_id": warning.warning_id,
        "annotator": "artifact_labeler_B",
        "labels": labels,
        "rationales": rationales,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/saw_bench_cs.jsonl")
    parser.add_argument("--out", default="annotation/artifact_labeler_B_pass.jsonl")
    args = parser.parse_args()

    rows = [label_warning(warning) for warning in load_warnings(args.data)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
    print(f"wrote {out}")
    print(f"warnings={len(rows)}")


if __name__ == "__main__":
    main()
