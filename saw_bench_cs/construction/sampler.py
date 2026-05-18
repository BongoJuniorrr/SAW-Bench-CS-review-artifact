"""Stratified sampling of warning records (paper §3.3, Table 5).

Stratification keys are (project, severity, category). Within each stratum we
sample without replacement using a deterministic RNG; the global target counts
match the paper's split sizes.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .spotbugs_runner import RawWarning
from .warning_filter import normalize_category


@dataclass
class SamplingTargets:
    train: int = 300
    validation: int = 90
    test: int = 90


def _project_to_split(project_to_split: dict[str, str]) -> dict[str, str]:
    return dict(project_to_split)


def stratified_sample(
    raw_warnings: Iterable[RawWarning],
    project_to_split: dict[str, str],
    targets: SamplingTargets = SamplingTargets(),
    seed: int = 7,
) -> list[RawWarning]:
    """Return the sampled warnings used as the benchmark population.

    The algorithm:
      1. Bucket warnings by (split, project, severity, category).
      2. For each split, compute per-project quotas proportional to the
         number of available warnings in that project, summing to the split
         target.
      3. Within each project, pull warnings round-robin across (severity,
         category) buckets until the project quota is met.

    The resulting sample is balanced across projects and exposes the category
    mix described in paper Table 5 ("Avg. snippets per category" range
    7.1--8.2 implies all categories must be represented across projects).
    """
    rng = random.Random(seed)
    project_to_split = _project_to_split(project_to_split)

    # split → project → list[RawWarning]
    pool: dict[str, dict[str, list[RawWarning]]] = defaultdict(lambda: defaultdict(list))
    for w in raw_warnings:
        split = project_to_split.get(w.project)
        if split is None:
            continue
        # Normalize category to the paper's vocabulary so stratification is consistent.
        normalized = RawWarning(**w.__dict__)
        normalized.category = normalize_category(w.rule_id, w.category)
        pool[split][w.project].append(normalized)

    sampled: list[RawWarning] = []
    target_map = {"train": targets.train, "validation": targets.validation, "test": targets.test}

    for split, target in target_map.items():
        per_project = pool[split]
        if not per_project:
            continue
        sizes = {p: len(rows) for p, rows in per_project.items()}
        total = sum(sizes.values()) or 1
        quotas = {p: max(1, math.floor(target * sizes[p] / total)) for p in sizes}
        leftover = target - sum(quotas.values())
        # distribute leftover slots to the projects with the most surplus
        for p in sorted(sizes, key=lambda x: -sizes[x]):
            if leftover <= 0:
                break
            quotas[p] += 1
            leftover -= 1

        for project, rows in per_project.items():
            quota = quotas.get(project, 0)
            if quota <= 0:
                continue
            # bucket by (severity, category) and round-robin
            buckets: dict[tuple[str, str], list[RawWarning]] = defaultdict(list)
            for r in rows:
                buckets[(r.severity, r.category)].append(r)
            for k in buckets:
                rng.shuffle(buckets[k])

            picked = 0
            keys = list(buckets)
            rng.shuffle(keys)
            while picked < quota and any(buckets.values()):
                progress = False
                for k in keys:
                    if not buckets[k]:
                        continue
                    sampled.append(buckets[k].pop())
                    picked += 1
                    progress = True
                    if picked >= quota:
                        break
                if not progress:
                    break

    return sampled
