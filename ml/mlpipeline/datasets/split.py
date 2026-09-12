"""Deterministic dataset splitting (Section 6.2).

Seeded + stratified by dominant class; grouped by source prefix (e.g. tile
prefix from the same survey image) to prevent adjacent-tile leakage.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field


class SplitResult(BaseModel):
    split_seed: int
    counts: dict[str, int] = Field(default_factory=dict)  # split -> #images
    per_split_classes: dict[str, dict[str, int]] = Field(default_factory=dict)
    grouped_by: str | None = None


def _group_key(image_rel_path: str, group_by_prefix_sep: str | None) -> str:
    """Grouping key for leakage prevention.

    With grouping, files sharing the prefix before `group_by_prefix_sep`
    (e.g. "surveyA_tile001" in "surveyA_tile001.png") stay in one split.
    """
    stem = Path(image_rel_path).stem
    if group_by_prefix_sep and group_by_prefix_sep in stem:
        return stem.split(group_by_prefix_sep)[0]
    return stem  # default: every file its own group


def split_dataset(
    image_paths: list[str],
    image_classes: dict[str, list[str]] | None,
    fractions: dict[str, float],
    seed: int = 42,
    group_by_prefix_sep: str | None = None,
) -> tuple[dict[str, list[str]], SplitResult]:
    """Assign images to splits deterministically.

    Args:
        image_paths: relative image paths (any order; sorted internally).
        image_classes: path -> class names present (for stratification).
        fractions: e.g. {"train": 0.7, "val": 0.2, "test": 0.1}; must sum to 1.
        seed: split seed (recorded in manifests for reproducibility).
        group_by_prefix_sep: keep files sharing a stem prefix in the same split.

    Returns:
        (split -> sorted list of image paths, SplitResult metadata)
    """
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {fractions}")
    splits = list(fractions.keys())

    groups: dict[str, list[str]] = defaultdict(list)
    for p in sorted(image_paths):
        groups[_group_key(p, group_by_prefix_sep)].append(p)

    rng = random.Random(seed)
    group_keys = sorted(groups.keys())
    rng.shuffle(group_keys)

    # Stratify by dominant class where possible (greedy round-robin over strata).
    def dominant(p_list: list[str]) -> str:
        if not image_classes:
            return "_none"
        c = Counter(cl for p in p_list for cl in image_classes.get(p, []))
        return c.most_common(1)[0][0] if c else "_none"

    strata: dict[str, list[str]] = defaultdict(list)
    for g in group_keys:
        strata[dominant(groups[g])].append(g)

    buckets: dict[str, list[str]] = {s: [] for s in splits}
    targets = {s: fractions[s] for s in splits}
    total = sum(len(groups[g]) for g in group_keys)
    counts = {s: 0 for s in splits}

    # Interleave strata so every split sees each class; fill respecting targets.
    for cls in sorted(strata):
        for g in strata[cls]:
            gsize = len(groups[g])
            # pick the split most under its target share
            def deficit(s: str) -> float:
                return targets[s] * total - counts[s]

            chosen = max(splits, key=deficit)
            buckets[chosen].extend(groups[g])
            counts[chosen] += gsize

    result = SplitResult(split_seed=seed, grouped_by=group_by_prefix_sep)
    out: dict[str, list[str]] = {}
    for s in splits:
        out[s] = sorted(buckets[s])
        result.counts[s] = len(out[s])
        result.per_split_classes[s] = dict(
            Counter(cl for p in out[s] for cl in (image_classes or {}).get(p, []))
        )
    return out, result


def write_yolo_split(
    interim_root: Path,
    processed_root: Path,
    assignment: dict[str, list[str]],
) -> Path:
    """Materialize a YOLO-format split dataset by copying files per assignment.

    ``assignment`` paths are relative to interim images/ (e.g. "images/surveyA_tile00.png"
    or flat "img00.png" — a leading "images/" component is stripped). writes
    processed_root/images/<split>/ and processed_root/labels/<split>/.
    """
    import shutil

    interim_root, processed_root = Path(interim_root), Path(processed_root)
    for split, paths in assignment.items():
        (processed_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (processed_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for rel in paths:
            rel = Path(rel)
            if rel.parts and rel.parts[0] == "images":
                rel = Path(*rel.parts[1:])
            src_img = interim_root / "images" / rel
            src_lbl = interim_root / "labels" / (rel.stem + ".txt")
            dst_img = processed_root / "images" / split / rel.name
            shutil.copy2(src_img, dst_img)
            if src_lbl.is_file():
                shutil.copy2(src_lbl, processed_root / "labels" / split / src_lbl.name)
    return processed_root
