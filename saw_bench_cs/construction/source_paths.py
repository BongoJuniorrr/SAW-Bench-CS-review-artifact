"""Resolve analyzer source paths to files inside a project checkout."""

from __future__ import annotations

from pathlib import Path


SOURCE_ROOTS = (
    Path("src/main/java"),
    Path("src/test/java"),
    Path("src/main/kotlin"),
    Path("src/test/kotlin"),
)


def resolve_source_file(repo_root: Path, source_path: str) -> Path:
    """Return the best checkout-local file for a SpotBugs source path.

    SpotBugs often reports Java package-relative paths such as
    `org/apache/Foo.java`; candidate extraction needs the actual checkout path,
    usually `src/main/java/org/apache/Foo.java`.
    """
    direct = repo_root / source_path
    if direct.is_file():
        return direct

    for root in SOURCE_ROOTS:
        candidate = repo_root / root / source_path
        if candidate.is_file():
            return candidate

    suffix_matches = list(repo_root.glob(f"**/{source_path}"))
    for match in suffix_matches:
        if match.is_file() and "target/" not in match.as_posix():
            return match
    return direct


def repo_relative(repo_root: Path, path: Path, fallback: str) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return fallback
