"""Filter generated, vendored, and trivial style-only warnings (paper §3.3)."""

from __future__ import annotations

import re
from typing import Iterable

from .spotbugs_runner import RawWarning


# Heuristics complement the SpotBugs exclude filter. They run on the
# normalized RawWarning rows.

GENERATED_PATTERNS = [
    re.compile(r"(^|/)target/generated-sources/"),
    re.compile(r"(^|/)build/generated/"),
    re.compile(r"(^|/)src/main/generated/"),
    re.compile(r"(^|/)gen/"),
]

VENDORED_PATTERNS = [
    re.compile(r"(^|/)third_party/"),
    re.compile(r"(^|/)vendor/"),
    re.compile(r"(^|/)external/"),
]

# Style-only SpotBugs categories that the paper excludes as "trivial".
STYLE_ONLY_CATEGORIES = {"STYLE", "I18N"}


# SpotBugs categories → paper categories (§4.3).
CATEGORY_MAP = {
    "CORRECTNESS": "Correctness",
    "BAD_PRACTICE": "Bad practice",
    "PERFORMANCE": "Performance",
    "SECURITY": "Security",
    "MT_CORRECTNESS": "Multithreaded correctness",
    "MALICIOUS_CODE": "Security",
    "EXPERIMENTAL": "Dodgy code",
}

# Sub-rule prefixes that override the SpotBugs category for the paper view.
RULE_PREFIX_OVERRIDES: list[tuple[str, str]] = [
    ("NP_", "Nullness"),
    ("OS_", "Resource management"),
    ("ODR_", "Resource management"),
    ("RV_", "Dodgy code"),
    ("EQ_", "Equality/hash-code"),
    ("HE_", "Equality/hash-code"),
]


def is_generated(path: str) -> bool:
    return any(p.search(path) for p in GENERATED_PATTERNS)


def is_vendored(path: str) -> bool:
    return any(p.search(path) for p in VENDORED_PATTERNS)


def is_style_only(category: str) -> bool:
    return category.upper() in STYLE_ONLY_CATEGORIES


def normalize_category(rule_id: str, raw_category: str) -> str:
    for prefix, cat in RULE_PREFIX_OVERRIDES:
        if rule_id.startswith(prefix):
            return cat
    return CATEGORY_MAP.get(raw_category.upper(), raw_category)


def keep_warning(w: RawWarning) -> bool:
    if not w.file or w.line < 1:
        return False
    if is_generated(w.file) or is_vendored(w.file):
        return False
    if is_style_only(w.category):
        return False
    return True


def filter_warnings(warnings: Iterable[RawWarning]) -> list[RawWarning]:
    return [w for w in warnings if keep_warning(w)]
