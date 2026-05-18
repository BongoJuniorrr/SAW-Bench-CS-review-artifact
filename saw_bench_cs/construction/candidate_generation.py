"""Candidate snippet generation for each warning (paper §3.4).

The 11 snippet types are produced by deterministic, lightweight extraction so
that construction is reproducible on commodity hardware. Snippet ordering is
stable: local types first, then cross-reference, then repository evidence.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

from ..schema import (
    MAX_SNIPPETS_PER_WARNING,
    MIN_SNIPPETS_PER_WARNING,
    CandidateSnippet,
    SnippetType,
)
from ..tokenization import count_tokens
from .source_paths import repo_relative, resolve_source_file
from .spotbugs_runner import RawWarning


# ----------------------------- helpers --------------------------------------

_METHOD_HEADER = re.compile(
    r"^\s*(public|private|protected|static|final|synchronized|abstract|\s)*"
    r"\s*[\w<>\[\],\s\?]+\s+(\w+)\s*\([^)]*\)\s*(throws[^{;]+)?\s*\{?\s*$"
)
_CLASS_HEADER = re.compile(r"^\s*(public|private|protected|abstract|final|\s)*\s*(class|interface|enum)\s+(\w+)")


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=False)


def _slice_method(lines: list[str], warning_line: int) -> tuple[int, int]:
    """Find the [start, end] line span (1-indexed) of the method enclosing warning_line."""
    if not lines or warning_line < 1 or warning_line > len(lines):
        return (1, len(lines) or 1)
    # walk upwards to find the method header
    start = 1
    for i in range(warning_line - 1, -1, -1):
        if _METHOD_HEADER.match(lines[i]):
            start = i + 1
            break
    # walk downwards counting braces
    depth = 0
    seen_open = False
    end = warning_line
    for i in range(start - 1, len(lines)):
        line = lines[i]
        for ch in line:
            if ch == "{":
                depth += 1
                seen_open = True
            elif ch == "}":
                depth -= 1
                if seen_open and depth == 0:
                    end = i + 1
                    return (start, end)
        end = i + 1
    return (start, end)


def _slice_class(lines: list[str], warning_line: int) -> tuple[int, int]:
    if not lines:
        return (1, 1)
    start = 1
    for i in range(warning_line - 1, -1, -1):
        if _CLASS_HEADER.match(lines[i]):
            start = i + 1
            break
    depth = 0
    seen_open = False
    end = len(lines)
    for i in range(start - 1, len(lines)):
        line = lines[i]
        for ch in line:
            if ch == "{":
                depth += 1
                seen_open = True
            elif ch == "}":
                depth -= 1
                if seen_open and depth == 0:
                    end = i + 1
                    return (start, end)
    return (start, end)


def _make_snippet(
    *,
    snippet_id: str,
    type_: SnippetType,
    path: str,
    line_start: int,
    line_end: int,
    text: str,
    generation_method: str,
) -> CandidateSnippet:
    return CandidateSnippet(
        snippet_id=snippet_id,
        type=type_,
        path=path,
        line_start=line_start,
        line_end=line_end,
        text=text,
        token_count=count_tokens(text),
        generation_method=generation_method,
    )


# ----------------------------- generators -----------------------------------

def gen_warning_line(idx: int, w: RawWarning, lines: list[str]) -> Optional[CandidateSnippet]:
    if w.line < 1 or w.line > len(lines):
        return None
    return _make_snippet(
        snippet_id=f"s{idx:02d}",
        type_="warning_line",
        path=w.file,
        line_start=w.line,
        line_end=w.line,
        text=lines[w.line - 1],
        generation_method="ast.warning_line",
    )


def gen_enclosing_method(idx: int, w: RawWarning, lines: list[str]) -> Optional[CandidateSnippet]:
    if not lines:
        return None
    s, e = _slice_method(lines, w.line)
    if e < s:
        return None
    return _make_snippet(
        snippet_id=f"s{idx:02d}",
        type_="enclosing_method",
        path=w.file,
        line_start=s, line_end=e,
        text="\n".join(lines[s - 1:e]),
        generation_method="ast.method_enclosing",
    )


def gen_enclosing_class(idx: int, w: RawWarning, lines: list[str]) -> Optional[CandidateSnippet]:
    if not lines:
        return None
    s, e = _slice_class(lines, w.line)
    if e < s:
        return None
    excerpt_lines = lines[s - 1:min(e, s + 80)]  # compact excerpt
    return _make_snippet(
        snippet_id=f"s{idx:02d}",
        type_="enclosing_class",
        path=w.file,
        line_start=s,
        line_end=s + len(excerpt_lines) - 1,
        text="\n".join(excerpt_lines),
        generation_method="ast.class_compact_excerpt",
    )


def _grep_callers(repo_root: Path, method_name: str, file_path: str) -> list[tuple[str, int, str]]:
    """Find files that mention `method_name(` outside `file_path`."""
    if not method_name:
        return []
    results: list[tuple[str, int, str]] = []
    pattern = re.compile(rf"\b{re.escape(method_name)}\s*\(")
    try:
        out = subprocess.check_output(
            ["grep", "-rn", "--include=*.java", f"{method_name}(", str(repo_root)],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="replace")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    for line in out.splitlines():
        try:
            file, line_no, snippet = line.split(":", 2)
        except ValueError:
            continue
        rel = str(Path(file).resolve().relative_to(repo_root.resolve()))
        if rel == file_path:
            continue
        if pattern.search(snippet):
            try:
                results.append((rel, int(line_no), snippet.strip()))
            except ValueError:
                continue
        if len(results) >= 3:
            break
    return results


def gen_callers(idx: int, w: RawWarning, repo_root: Path, method_name: str) -> list[CandidateSnippet]:
    snippets = []
    for rel, line_no, text in _grep_callers(repo_root, method_name, w.file):
        snippets.append(_make_snippet(
            snippet_id=f"s{idx:02d}",
            type_="caller",
            path=rel,
            line_start=line_no, line_end=line_no,
            text=text,
            generation_method="grep.method_call_site",
        ))
        idx += 1
    return snippets


_INVOCATION = re.compile(r"\b(\w+)\s*\(")


def gen_callees(idx: int, w: RawWarning, lines: list[str], method_span: tuple[int, int]) -> list[CandidateSnippet]:
    """Pick up to two distinct callee lines inside the enclosing method."""
    s, e = method_span
    snippets: list[CandidateSnippet] = []
    seen: set[str] = set()
    for ln in range(max(1, w.line - 5), min(e, w.line + 10) + 1):
        if ln < 1 or ln > len(lines):
            continue
        m = _INVOCATION.search(lines[ln - 1])
        if not m:
            continue
        name = m.group(1)
        if name in seen or name in {"if", "for", "while", "switch", "return", "new"}:
            continue
        seen.add(name)
        snippets.append(_make_snippet(
            snippet_id=f"s{idx:02d}",
            type_="callee",
            path=w.file,
            line_start=ln, line_end=ln,
            text=lines[ln - 1],
            generation_method="ast.invocation_in_method",
        ))
        idx += 1
        if len(snippets) >= 2:
            break
    return snippets


_FIELD_DECL = re.compile(r"^\s*(private|protected|public)\s+[\w<>\[\],\s\?]+\s+\w+\s*[=;]")


def gen_field_declarations(idx: int, w: RawWarning, lines: list[str], class_span: tuple[int, int]) -> list[CandidateSnippet]:
    s, e = class_span
    out = []
    for ln in range(s, min(e, s + 60) + 1):
        if ln < 1 or ln > len(lines):
            continue
        if _FIELD_DECL.match(lines[ln - 1]):
            out.append(_make_snippet(
                snippet_id=f"s{idx:02d}",
                type_="field_or_type_declaration",
                path=w.file,
                line_start=ln, line_end=ln,
                text=lines[ln - 1],
                generation_method="ast.class_field_decl",
            ))
            idx += 1
        if len(out) >= 1:
            break
    return out


_ANNOTATION = re.compile(r"@(Nullable|NonNull|Nonnull|Override|Deprecated|GuardedBy|CheckReturnValue|SuppressWarnings)\b")


def gen_annotations(idx: int, w: RawWarning, lines: list[str], method_span: tuple[int, int]) -> list[CandidateSnippet]:
    s, e = method_span
    out = []
    for ln in range(max(1, s - 3), min(e, s + 5) + 1):
        if ln < 1 or ln > len(lines):
            continue
        if _ANNOTATION.search(lines[ln - 1]):
            out.append(_make_snippet(
                snippet_id=f"s{idx:02d}",
                type_="annotation_or_contract",
                path=w.file,
                line_start=ln, line_end=ln,
                text=lines[ln - 1],
                generation_method="regex.annotation_near_method",
            ))
            idx += 1
            if len(out) >= 1:
                break
    return out


def gen_tests(idx: int, w: RawWarning, repo_root: Path, class_name: str) -> list[CandidateSnippet]:
    """Tests under src/test/** whose paths mention the class or its short name."""
    if not class_name:
        return []
    short = class_name.rsplit(".", 1)[-1]
    test_root = repo_root / "src" / "test" / "java"
    if not test_root.is_dir():
        return []
    matches: list[CandidateSnippet] = []
    for path in test_root.rglob("*.java"):
        if short in path.stem:
            try:
                rel = str(path.resolve().relative_to(repo_root.resolve()))
            except ValueError:
                continue
            text = path.read_text(encoding="utf-8", errors="replace").splitlines()
            head = "\n".join(text[:20])
            matches.append(_make_snippet(
                snippet_id=f"s{idx:02d}",
                type_="test",
                path=rel,
                line_start=1, line_end=min(len(text), 20),
                text=head,
                generation_method="fs.test_by_class_name",
            ))
            idx += 1
            if len(matches) >= 1:
                break
    return matches


def gen_configuration(idx: int, w: RawWarning, repo_root: Path) -> list[CandidateSnippet]:
    """Surface SpotBugs filter or build snippets that match the rule_id."""
    candidates = [
        repo_root / "spotbugs-exclude.xml",
        repo_root / "spotbugs-include.xml",
        repo_root / "pom.xml",
        repo_root / "build.gradle",
        repo_root / "build.gradle.kts",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if w.rule_id and w.rule_id in text:
            window_start = max(0, text.find(w.rule_id) - 80)
            excerpt = text[window_start: window_start + 240]
            return [_make_snippet(
                snippet_id=f"s{idx:02d}",
                type_="configuration",
                path=str(path.relative_to(repo_root)),
                line_start=1, line_end=1,
                text=excerpt,
                generation_method="fs.config_rule_match",
            )]
    return []


def gen_similar_code(idx: int, w: RawWarning, lines: list[str], method_span: tuple[int, int], repo_root: Path) -> list[CandidateSnippet]:
    """Top-1 similar method body in the same project by token Jaccard."""
    s, e = method_span
    if e - s < 2:
        return []
    target = "\n".join(lines[s - 1:e])
    target_tokens = set(re.findall(r"\w+", target))
    if not target_tokens:
        return []

    best: tuple[float, Path, int, int, str] | None = None
    for path in (repo_root / "src" / "main" / "java").rglob("*.java"):
        if str(path) == str(repo_root / w.file):
            continue
        try:
            file_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        if len(file_lines) < 5:
            continue
        # cheap method scan: take 8-line windows
        for start_line in range(0, len(file_lines), 25):
            window = file_lines[start_line:start_line + 12]
            if len(window) < 6:
                continue
            tokens = set(re.findall(r"\w+", "\n".join(window)))
            if not tokens:
                continue
            jaccard = len(target_tokens & tokens) / len(target_tokens | tokens)
            if jaccard > 0.25 and (best is None or jaccard > best[0]):
                best = (
                    jaccard,
                    path,
                    start_line + 1,
                    start_line + len(window),
                    "\n".join(window),
                )
        # cap repo scan; keep construction cost bounded.
    if best is None:
        return []
    _, path, ls, le, text = best
    rel = str(path.resolve().relative_to(repo_root.resolve()))
    return [_make_snippet(
        snippet_id=f"s{idx:02d}",
        type_="similar_code",
        path=rel,
        line_start=ls, line_end=le,
        text=text,
        generation_method="jaccard.same_project",
    )]


def gen_history(idx: int, w: RawWarning, repo_root: Path) -> list[CandidateSnippet]:
    """`git blame` for the warning line + last commit subject for the file."""
    try:
        blame = subprocess.check_output(
            ["git", "-C", str(repo_root), "blame", "-L", f"{w.line},{w.line}",
             "--line-porcelain", w.file],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="replace")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    summary_match = re.search(r"^summary (.+)$", blame, re.MULTILINE)
    author_match = re.search(r"^author (.+)$", blame, re.MULTILINE)
    if summary_match is None:
        return []
    text = f"{summary_match.group(1)} (by {author_match.group(1) if author_match else 'unknown'})"
    return [_make_snippet(
        snippet_id=f"s{idx:02d}",
        type_="history_or_diff",
        path=w.file,
        line_start=w.line, line_end=w.line,
        text=text,
        generation_method="git.blame_porcelain",
    )]


# ----------------------------- top-level ------------------------------------

def generate_candidates(w: RawWarning, repo_root: Path) -> list[CandidateSnippet]:
    """Generate the candidate set for one warning, capped at 10 (paper §3.4)."""
    file_path = resolve_source_file(repo_root, w.file)
    w = replace(w, file=repo_relative(repo_root, file_path, w.file))
    lines = _read_lines(file_path)
    method_span = _slice_method(lines, w.line)
    class_span = _slice_class(lines, w.line)
    method_name = ""
    # parse the method name from the header line
    if method_span[0] >= 1 and method_span[0] <= len(lines):
        m = _METHOD_HEADER.match(lines[method_span[0] - 1])
        if m:
            method_name = m.group(2)

    snippets: list[CandidateSnippet] = []
    idx = 1

    def push(items: Iterable[CandidateSnippet]) -> None:
        nonlocal idx
        for s in items:
            if not s:
                continue
            if len(snippets) >= MAX_SNIPPETS_PER_WARNING:
                return
            snippets.append(s)
            idx += 1

    one = lambda s: [s] if s else []

    push(one(gen_warning_line(idx, w, lines)))
    push(one(gen_enclosing_method(idx, w, lines)))
    push(one(gen_enclosing_class(idx, w, lines)))
    push(gen_callers(idx, w, repo_root, method_name))
    push(gen_callees(idx, w, lines, method_span))
    push(gen_field_declarations(idx, w, lines, class_span))
    push(gen_annotations(idx, w, lines, method_span))
    push(gen_tests(idx, w, repo_root, w.class_name))
    push(gen_configuration(idx, w, repo_root))
    push(gen_similar_code(idx, w, lines, method_span, repo_root))
    push(gen_history(idx, w, repo_root))

    # Deduplicate by (path, line_start, line_end) keeping first.
    seen = set()
    deduped: list[CandidateSnippet] = []
    for s in snippets:
        key = (s.path, s.line_start, s.line_end)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)

    if len(deduped) < MIN_SNIPPETS_PER_WARNING:
        return deduped  # caller surfaces the shortfall via consistency_checks
    return deduped[:MAX_SNIPPETS_PER_WARNING]
