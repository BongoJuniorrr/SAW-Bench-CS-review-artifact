"""Six-stage construction pipeline (paper §3.2)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from ..io import save_warnings
from ..schema import (
    MIN_SNIPPETS_PER_WARNING,
    RelevanceLabel,
    Warning,
    WarningContext,
)
from .annotation import AnnotatorPass, merge_passes
from .candidate_generation import generate_candidates
from .consistency_checks import validate_dataset
from .projects import ProjectsConfig, build_project, checkout, load_projects_config
from .sampler import SamplingTargets, stratified_sample
from .source_paths import resolve_source_file
from .spotbugs_runner import RawWarning, parse_report, run_spotbugs
from .splits import apply_splits, assert_disjoint, load_split_assignment
from .warning_filter import filter_warnings, normalize_category


# Stage ordering matches paper §3.2.
STAGES = (
    "select_projects",
    "run_spotbugs",
    "filter_warnings",
    "sample_warnings",
    "generate_candidates",
    "assign_splits",
)


def warning_id(spec_name: str, idx: int) -> str:
    return f"{spec_name}#{idx:04d}"


def build_warning_context(repo_root: Path, raw: RawWarning) -> WarningContext:
    file = resolve_source_file(repo_root, raw.file)
    if not file.is_file():
        return WarningContext(warning_line="", complete_statement="", method_excerpt="")
    lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
    if 1 <= raw.line <= len(lines):
        line_text = lines[raw.line - 1]
    else:
        line_text = ""
    # complete_statement: simple extension to next ';' or '{'/'}'
    statement = line_text
    for ln in range(raw.line, min(len(lines), raw.line + 8)):
        statement += "\n" + lines[ln]
        if any(ch in lines[ln] for ch in (";", "{", "}")):
            break
    method_excerpt = "\n".join(
        lines[max(0, raw.line - 5):min(len(lines), raw.line + 5)]
    )
    return WarningContext(
        warning_line=line_text,
        complete_statement=statement,
        method_excerpt=method_excerpt,
    )


def construct_warning(
    raw: RawWarning,
    repo_root: Path,
    counter: int,
) -> Warning:
    candidates = generate_candidates(raw, repo_root)
    return Warning(
        warning_id=warning_id(raw.project, counter),
        project=raw.project,
        commit="",  # filled in by caller from project spec
        tool="spotbugs",
        rule_id=raw.rule_id,
        category=normalize_category(raw.rule_id, raw.category),
        severity=raw.severity,
        file=raw.file,
        line=raw.line,
        warning_message=raw.long_message or raw.message,
        warning_context=build_warning_context(repo_root, raw),
        candidate_snippets=candidates,
        labels=[],  # populated post-annotation
        split="train",  # overridden by apply_splits
    )


def attach_labels(
    warnings: list[Warning],
    passes_by_warning: dict[str, list[AnnotatorPass]],
) -> list[Warning]:
    for w in warnings:
        passes = passes_by_warning.get(w.warning_id, [])
        merged = merge_passes(passes)
        # Drop labels that don't reference any candidate (e.g., snippets removed
        # during dedup) so the validator passes.
        sids = {s.snippet_id for s in w.candidate_snippets}
        w.labels = [lbl for lbl in merged if lbl.snippet_id in sids]
    return warnings


def run_pipeline(
    *,
    config_path: str | Path,
    splits_path: str | Path,
    work_root: str | Path,
    out_path: str | Path,
    annotation_passes_path: str | Path | None = None,
    sampling_targets: SamplingTargets = SamplingTargets(),
    seed: int = 7,
    build_projects: bool = True,
    require_labels: bool = True,
    drop_short_candidates: bool = False,
    continue_on_project_error: bool = False,
) -> Path:
    config: ProjectsConfig = load_projects_config(config_path)
    split_assignment = load_split_assignment(splits_path)
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    project_errors: dict[str, str] = {}

    # Stage 1: select_projects (clone + checkout).
    project_roots: dict[str, Path] = {}
    for spec in config.projects:
        try:
            project_roots[spec.name] = checkout(spec, work_root / "checkouts")
        except Exception as exc:
            if not continue_on_project_error:
                raise
            project_errors[spec.name] = f"checkout failed: {exc}"

    # Stage 1b: compile projects so SpotBugs has bytecode to analyze.
    if build_projects:
        for spec in config.projects:
            if spec.name not in project_roots:
                continue
            try:
                build_project(spec, project_roots[spec.name])
            except Exception as exc:
                if not continue_on_project_error:
                    raise
                project_errors[spec.name] = f"build failed: {exc}"
                project_roots.pop(spec.name, None)

    # Stage 2: run_spotbugs.
    raw_by_project: dict[str, list[RawWarning]] = {}
    spotbugs_dir = work_root / "spotbugs"
    for spec in config.projects:
        root = project_roots.get(spec.name)
        if root is None:
            continue
        try:
            report = run_spotbugs(spec, root, config, spotbugs_dir)
            raw_by_project[spec.name] = parse_report(report, spec.name)
        except Exception as exc:
            if not continue_on_project_error:
                raise
            project_errors[spec.name] = f"spotbugs failed: {exc}"

    if project_errors:
        errors_path = work_root / "project_errors.txt"
        errors_path.parent.mkdir(parents=True, exist_ok=True)
        errors_path.write_text(
            "\n".join(
                f"{name}: {message}"
                for name, message in sorted(project_errors.items())
            ),
            encoding="utf-8",
        )

    # Stage 3: filter_warnings.
    raw_filtered: list[RawWarning] = []
    for project, rows in raw_by_project.items():
        raw_filtered.extend(filter_warnings(rows))

    # Stage 4: sample_warnings.
    sampled = stratified_sample(
        raw_filtered, project_to_split=split_assignment, targets=sampling_targets, seed=seed,
    )

    # Stage 5: generate_candidates.
    project_counters: dict[str, int] = {}
    warnings: list[Warning] = []
    for raw in sampled:
        idx = project_counters.get(raw.project, 0) + 1
        project_counters[raw.project] = idx
        w = construct_warning(raw, project_roots[raw.project], idx)
        # commit pin from config
        spec = next(s for s in config.projects if s.name == raw.project)
        w.commit = spec.commit
        if (
            drop_short_candidates
            and len(w.candidate_snippets) < MIN_SNIPPETS_PER_WARNING
        ):
            continue
        warnings.append(w)

    # Stage 6: assign_splits.
    warnings = apply_splits(warnings, split_assignment)
    assert_disjoint(warnings)

    # Annotation labels (loaded if present).
    if annotation_passes_path is not None and Path(annotation_passes_path).is_file():
        from .annotation import load_passes  # local import keeps cycle small
        passes = load_passes(annotation_passes_path)
        passes_by_warning: dict[str, list[AnnotatorPass]] = {}
        for ap in passes:
            passes_by_warning.setdefault(ap.warning_id, []).append(ap)
        warnings = attach_labels(warnings, passes_by_warning)

    # Validate and write.
    report = validate_dataset(warnings, require_labels=require_labels)
    if not report.passed:
        raise RuntimeError(f"Dataset failed validation:\n{report.summary()}")

    out_path = Path(out_path)
    save_warnings(warnings, out_path)
    return out_path
