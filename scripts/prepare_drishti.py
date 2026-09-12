#!/usr/bin/env python3
"""Prepare DRISHTI-SSS dataset for training.

Remaps class IDs from DRISHTI's {1,2,3,4} (excluding crab_pot=0)
to our {0,1,2,3} and creates a YOLO data.yaml for ultralytics.

DRISHTI mapping:
  0: crab_pot  (excluded — no examples)
  1: submarine_pipeline  -> 0
  2: shipwreck           -> 1
  3: ghost_net           -> 2
  4: mine_cylinder       -> 3
"""
from __future__ import annotations

import shutil
from pathlib import Path

# DRISHTI class ID -> our class ID
REMAPPING = {
    1: 0,  # submarine_pipeline
    2: 1,  # shipwreck
    3: 2,  # ghost_net
    4: 3,  # mine_cylinder
}

CLASS_NAMES = {
    0: "submarine_pipeline",
    1: "shipwreck",
    2: "ghost_net",
    3: "mine_cylinder",
}

SOURCE = Path(__file__).resolve().parents[1] / "datasets" / "raw" / "drishti-sss"
TARGET = Path(__file__).resolve().parents[1] / "datasets" / "processed" / "drishti-sss"


def remap_label(src_label: Path, dst_label: Path) -> dict:
    """Read a YOLO label, remap class IDs, write to dst. Return stats."""
    stats = {"lines": 0, "remapped": 0, "skipped": 0}
    lines = []
    for line in src_label.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        old_id = int(parts[0])
        if old_id not in REMAPPING:
            stats["skipped"] += 1
            continue
        new_id = REMAPPING[old_id]
        parts[0] = str(new_id)
        lines.append(" ".join(parts))
        stats["lines"] += 1
        stats["remapped"] += 1
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return stats


def prepare() -> None:
    if not SOURCE.is_dir():
        raise SystemExit(f"Source not found: {SOURCE}")

    total_stats = {"images": 0, "labels": 0, "annotations": 0, "skipped": 0}

    for split in ["train", "val", "test"]:
        src_images = SOURCE / split / "images"
        src_labels = SOURCE / split / "labels"
        dst_images = TARGET / split / "images"
        dst_labels = TARGET / split / "labels"

        if not src_images.is_dir():
            print(f"  Skipping {split}: no images dir")
            continue

        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        # Copy images (don't remap — filenames are fine)
        for img_file in sorted(src_images.iterdir()):
            if img_file.is_file():
                shutil.copy2(img_file, dst_images / img_file.name)
                total_stats["images"] += 1

        # Remap labels
        if src_labels.is_dir():
            for lbl_file in sorted(src_labels.iterdir()):
                if lbl_file.is_file() and lbl_file.suffix == ".txt":
                    dst_lbl = dst_labels / lbl_file.name
                    stats = remap_label(lbl_file, dst_lbl)
                    total_stats["labels"] += 1
                    total_stats["annotations"] += stats["lines"]
                    total_stats["skipped"] += stats["skipped"]

        print(f"  {split}: {total_stats['images']} images copied, labels remapped")

    # Copy drishti.yaml for reference
    src_yaml = SOURCE / "drishti.yaml"
    if src_yaml.exists():
        shutil.copy2(src_yaml, TARGET / "drishti_original.yaml")

    # Write our data.yaml
    data_yaml = TARGET / "data.yaml"
    import yaml
    content = {
        "path": str(TARGET.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 4,
        "names": CLASS_NAMES,
    }
    data_yaml.write_text(yaml.dump(content, sort_keys=False), encoding="utf-8")

    print(f"\nDone. Total: {total_stats['images']} images, "
          f"{total_stats['annotations']} annotations, "
          f"{total_stats['skipped']} skipped (crab_pot)")
    print(f"Output: {TARGET}")
    print(f"Data YAML: {data_yaml}")


if __name__ == "__main__":
    prepare()
