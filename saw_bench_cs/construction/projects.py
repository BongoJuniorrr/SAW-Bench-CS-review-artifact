"""Project metadata loading, checkout, and build helpers (paper §3.3)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class ProjectSpec:
    name: str
    repo: str
    commit: str
    build: str  # 'maven' | 'maven-fast' | 'gradle'
    split: str  # informational; authoritative split lives in splits.yaml


@dataclass
class ProjectsConfig:
    spotbugs_version: str
    spotbugs_effort: str
    spotbugs_threshold: str
    exclude_filter: str
    java_version: str
    projects: list[ProjectSpec]


def load_projects_config(path: str | Path) -> ProjectsConfig:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    spotbugs = raw["spotbugs"]
    defaults = raw["defaults"]
    projects = [ProjectSpec(**p) for p in raw["projects"]]
    return ProjectsConfig(
        spotbugs_version=spotbugs["version"],
        spotbugs_effort=spotbugs["effort"],
        spotbugs_threshold=spotbugs["threshold"],
        exclude_filter=spotbugs["exclude_filter"],
        java_version=defaults["java_version"],
        projects=projects,
    )


def iter_projects(config: ProjectsConfig) -> Iterator[ProjectSpec]:
    yield from config.projects


def checkout(spec: ProjectSpec, work_root: Path) -> Path:
    """Clone (if needed) and pin the project to its target commit.

    Returns the absolute path to the checkout. Idempotent so the pipeline can be
    resumed; raises CalledProcessError if git fails.
    """
    work_root.mkdir(parents=True, exist_ok=True)
    target = work_root / spec.name
    if not target.exists():
        subprocess.check_call(["git", "clone", spec.repo, str(target)])
    subprocess.check_call(["git", "-C", str(target), "fetch", "--tags", "--force"])
    subprocess.check_call(["git", "-C", str(target), "checkout", "--force", spec.commit])
    return target


def build_project(spec: ProjectSpec, project_root: Path) -> None:
    """Compile a checkout so SpotBugs can analyze bytecode.

    SpotBugs requires class files, so source checkout alone is not enough for
    the construction pipeline. This helper uses the project's configured build
    system and prefers repository wrappers when available.
    """
    if spec.build == "maven":
        executable = "./mvnw" if (project_root / "mvnw").is_file() else "mvn"
        cmd = [executable, "-DskipTests", "-DskipITs", "package"]
    elif spec.build == "maven-fast":
        executable = "./mvnw" if (project_root / "mvnw").is_file() else "mvn"
        cmd = [
            executable,
            "-DskipTests",
            "-DskipITs",
            "-Dmaven.javadoc.skip=true",
            "-Dspotbugs.skip=true",
            "-Dcheckstyle.skip=true",
            "-Drat.skip=true",
            "-Dlicense.skip=true",
            "-Drewrite.skip=true",
            "-Dcyclonedx.skip=true",
            "-Dfmt.skip=true",
            "-Dimpsort.skip=true",
            "-Denforcer.skip=true",
            "compile",
        ]
    elif spec.build == "gradle":
        executable = "./gradlew" if (project_root / "gradlew").is_file() else "gradle"
        cmd = [executable, "assemble", "-x", "test"]
    else:
        raise ValueError(f"Unsupported build system for {spec.name}: {spec.build!r}")

    subprocess.check_call(cmd, cwd=project_root)
