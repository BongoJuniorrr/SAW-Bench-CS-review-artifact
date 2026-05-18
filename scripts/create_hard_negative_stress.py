"""Hard-negative stress test: inject plausible but irrelevant snippets to test ranker robustness."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import _bootstrap  # noqa: F401
from saw_bench_cs.io import load_warnings
from saw_bench_cs.schema import RelevanceLabel


def _tokenize_simple(text: str) -> set[str]:
    """Simple tokenization for similarity."""
    return set(text.lower().split())


def _jaccard_similarity(text1: str, text2: str) -> float:
    """Jaccard similarity between two texts."""
    t1 = _tokenize_simple(text1)
    t2 = _tokenize_simple(text2)
    if not t1 | t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def generate_hard_negatives(
    warnings: list,
    num_per_warning: int = 5,
    seed: int = 7,
) -> tuple[list, dict]:
    """Generate hard negative candidates for each warning.
    
    Returns:
        Modified warnings with injected hard negatives (marked irrelevant)
        Dictionary with statistics about injected negatives
    """
    random.seed(seed)
    
    # Index warnings by project, category, rule_id
    by_project = defaultdict(list)
    by_category = defaultdict(list)
    by_rule = defaultdict(list)
    
    for w in warnings:
        by_project[w.project].append(w)
        by_category[w.category].append(w)
        by_rule[w.rule_id].append(w)
    
    # Snippet ID counter for hard negatives
    max_snippet_num = 0
    for w in warnings:
        for s in w.candidate_snippets:
            try:
                num = int(s.snippet_id[1:])  # e.g., "s01" -> 1
                max_snippet_num = max(max_snippet_num, num)
            except (ValueError, IndexError):
                pass
    
    stats = {
        "total_injected": 0,
        "avg_per_warning": 0.0,
        "distribution": {
            "same_project": 0,
            "same_category": 0,
            "same_rule": 0,
            "high_overlap": 0,
        },
    }
    
    modified_warnings = []
    next_snippet_num = max_snippet_num + 1
    
    for w in warnings:
        candidate_sources = []
        
        # Collect sources from same project (exclude current warning)
        for other in by_project.get(w.project, []):
            if other.warning_id != w.warning_id:
                candidate_sources.extend([
                    (s, "same_project") for s in other.candidate_snippets
                    if s.snippet_id not in {cs.snippet_id for cs in w.candidate_snippets}
                ])
        
        # Collect sources from same category
        for other in by_category.get(w.category, []):
            if other.warning_id != w.warning_id:
                candidate_sources.extend([
                    (s, "same_category") for s in other.candidate_snippets
                    if s.snippet_id not in {cs.snippet_id for cs in w.candidate_snippets}
                ])
        
        # Collect sources from same rule
        for other in by_rule.get(w.rule_id, []):
            if other.warning_id != w.warning_id:
                candidate_sources.extend([
                    (s, "same_rule") for s in other.candidate_snippets
                    if s.snippet_id not in {cs.snippet_id for cs in w.candidate_snippets}
                ])
        
        # Filter by jaccard similarity to warning message
        high_overlap = [
            (s, "high_overlap") for s, src_type in candidate_sources
            if _jaccard_similarity(w.warning_message, s.text) > 0.3 and
            src_type == "same_rule"
        ]
        candidate_sources.extend(high_overlap)
        
        if not candidate_sources:
            modified_warnings.append(w)
            continue
        
        # Select up to num_per_warning hard negatives
        sampled = random.sample(candidate_sources, min(num_per_warning, len(candidate_sources)))
        
        # Create new snippets for hard negatives (don't reuse snippet IDs)
        injected_snippets = []
        injected_labels = []
        
        for snippet, source_type in sampled:
            new_snippet_id = f"hn{next_snippet_num:02d}"
            next_snippet_num += 1
            
            # Copy snippet with new ID
            new_snippet = replace(snippet, snippet_id=new_snippet_id)
            injected_snippets.append(new_snippet)
            
            # Add irrelevant label
            injected_labels.append(
                RelevanceLabel(snippet_id=new_snippet_id, relevance="irrelevant", annotator_count=1)
            )
            
            stats["distribution"][source_type] += 1
        
        # Create modified warning
        w_new = replace(
            w,
            candidate_snippets=w.candidate_snippets + injected_snippets,
            labels=w.labels + injected_labels,
        )
        modified_warnings.append(w_new)
        stats["total_injected"] += len(injected_labels)
    
    stats["avg_per_warning"] = stats["total_injected"] / len(warnings) if warnings else 0
    
    return modified_warnings, stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/saw_bench_cs.jsonl")
    p.add_argument("--num-per-warning", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="data/saw_bench_cs_hardneg.jsonl")
    args = p.parse_args()
    
    # Load warnings
    warnings = load_warnings(args.data)
    
    # Generate hard negatives
    print(f"Generating up to {args.num_per_warning} hard negatives per warning...")
    modified, stats = generate_hard_negatives(warnings, args.num_per_warning, args.seed)
    
    print(f"  Total injected: {stats['total_injected']}")
    print(f"  Average per warning: {stats['avg_per_warning']:.2f}")
    print(f"  Distribution:")
    for source_type, count in stats['distribution'].items():
        print(f"    {source_type}: {count}")
    
    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        for w in modified:
            fh.write(json.dumps(w.to_dict()) + "\n")
    
    print(f"\nWrote {out_path} with {len(modified)} warnings")


if __name__ == "__main__":
    main()
