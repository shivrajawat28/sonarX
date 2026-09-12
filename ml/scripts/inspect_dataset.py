"""Inspect a YOLO-format dataset: stats + validation report + class distribution.

Phase 3 starting point. Usage:
    python -m ml.scripts.inspect_dataset --root datasets/processed/<name> \
        --name <name> --classes alpha,beta --save-manifest datasets/manifests/<name>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mlpipeline.datasets import build_manifest, save_manifest, validate_dataset


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="dataset root (images/ + labels/ layout)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--classes", default="", help="comma-separated class names by id order")
    ap.add_argument("--format", default="yolo")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--save-manifest", default="", help="path to write the manifest JSON")
    ap.add_argument("--no-image-decode", action="store_true", help="skip decodability check")
    args = ap.parse_args(argv)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    root = Path(args.root)

    print(f"== inspect_dataset: {args.name} @ {root} ==")
    try:
        manifest = build_manifest(
            root, name=args.name, format=args.format,
            class_names=classes or None, split_seed=args.split_seed,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(f"images: {manifest.total_images}  instances: {manifest.total_instances}")
    print(f"per-split: {json.dumps(manifest.stats.get('per_split', {}))}")
    print(f"class counts: {json.dumps(manifest.stats.get('class_counts', {}), indent=2)}")
    print(f"images without labels: {manifest.stats.get('images_without_labels', 0)}")

    report = validate_dataset(manifest, check_images_decodable=not args.no_image_decode)
    print(f"\nvalidation: {'OK' if report.ok else 'FAILED'}")
    for issue in report.errors:
        print(f"  [error]   {issue.code}: {issue.detail}")
    for issue in report.warnings:
        print(f"  [warning] {issue.code}: {issue.detail}")

    if args.save_manifest:
        out = save_manifest(manifest, Path(args.save_manifest))
        print(f"\nmanifest written: {out}")

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
