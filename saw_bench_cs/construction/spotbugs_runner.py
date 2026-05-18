"""Run SpotBugs against a project checkout and parse warning records.

The pipeline assumes the project has been built (Maven `package` or Gradle
`assemble`) so that compiled bytecode is available under typical output dirs.
For full reproducibility on commodity hardware, the runner shells out to
SpotBugs in `-xml:withMessages` mode and parses the resulting report.
"""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .projects import ProjectSpec, ProjectsConfig


@dataclass
class RawWarning:
    """Lightly normalized SpotBugs report row."""

    project: str
    rule_id: str          # e.g. NP_NULL_ON_SOME_PATH
    category: str         # e.g. CORRECTNESS
    severity: str         # 'high' | 'medium' | 'low'
    file: str             # repo-relative path
    line: int
    message: str          # short SpotBugs message
    long_message: str     # long-form SpotBugs message
    class_name: str       # fully-qualified class name


_PRIORITY_TO_SEVERITY = {"1": "high", "2": "medium", "3": "low"}


def find_class_files(project_root: Path) -> list[Path]:
    """Locate compiled class file roots for SpotBugs (`-classpath` arg)."""
    candidates = [
        project_root / "target" / "classes",
        project_root / "build" / "classes" / "java" / "main",
        project_root / "out" / "production" / "classes",
    ]
    return [c for c in candidates if c.is_dir()]


def find_source_dirs(project_root: Path) -> list[Path]:
    return [d for d in [
        project_root / "src" / "main" / "java",
        project_root / "src" / "main" / "kotlin",
    ] if d.is_dir()]


def run_spotbugs(
    spec: ProjectSpec,
    project_root: Path,
    config: ProjectsConfig,
    out_dir: Path,
) -> Path:
    """Invoke SpotBugs and return the path to the XML report.

    Caller is expected to ensure SpotBugs is installed. The exact CLI is
    parameterized so the pipeline can be redirected to a containerized run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"{spec.name}.spotbugs.xml"

    classpath_dirs = find_class_files(project_root)
    if not classpath_dirs:
        raise RuntimeError(
            f"No compiled classes found under {project_root}. Build the "
            "project (e.g. `mvn -DskipTests package` or `./gradlew assemble`) "
            "before running SpotBugs."
        )

    cmd = [
        "spotbugs",
        "-textui",
        "-xml:withMessages",
        "-effort:" + config.spotbugs_effort,
        "-" + config.spotbugs_threshold,
        "-exclude", config.exclude_filter,
        "-output", str(report),
    ] + [str(p) for p in classpath_dirs]

    subprocess.check_call(cmd)
    return report


def parse_report(report: Path, project: str) -> list[RawWarning]:
    """Parse a `spotbugs -xml:withMessages` report into RawWarning records."""
    tree = ET.parse(report)
    root = tree.getroot()
    warnings: list[RawWarning] = []
    for bug in root.findall("BugInstance"):
        rule = bug.attrib.get("type", "")
        category = bug.attrib.get("category", "")
        priority = bug.attrib.get("priority", "2")
        severity = _PRIORITY_TO_SEVERITY.get(priority, "medium")
        message_el = bug.find("LongMessage")
        long_message = message_el.text.strip() if (message_el is not None and message_el.text) else ""
        short_el = bug.find("ShortMessage")
        short_message = short_el.text.strip() if (short_el is not None and short_el.text) else long_message
        src = bug.find("SourceLine")
        if src is None:
            continue
        file_path = src.attrib.get("sourcepath") or src.attrib.get("relSourcepath", "")
        try:
            line = int(src.attrib.get("start", "0"))
        except ValueError:
            line = 0
        cls = bug.find("Class")
        class_name = cls.attrib.get("classname", "") if cls is not None else ""
        warnings.append(RawWarning(
            project=project,
            rule_id=rule,
            category=category,
            severity=severity,
            file=file_path,
            line=line,
            message=short_message,
            long_message=long_message,
            class_name=class_name,
        ))
    return warnings


def run_all(
    config: ProjectsConfig,
    project_roots: dict[str, Path],
    out_dir: Path,
) -> dict[str, list[RawWarning]]:
    """Run SpotBugs over every project in the config; return per-project warnings."""
    results: dict[str, list[RawWarning]] = {}
    for spec in config.projects:
        root = project_roots[spec.name]
        report = run_spotbugs(spec, root, config, out_dir)
        results[spec.name] = parse_report(report, spec.name)
    return results
