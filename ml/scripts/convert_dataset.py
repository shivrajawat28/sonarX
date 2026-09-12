"""Convert an interim dataset into a split YOLO-format dataset + manifest.

Pipeline: interim (flat images/labels) -> validate -> stratified split ->
processed root -> manifest JSON. Usage:
    python -m ml.scripts.convert_dataset --interim datasets/interim/<name> \
        --out datasets/processed/<name> --name <name> --classes alpha,beta \
        --group-sep _tile --save-manifest datasets/manifests/<name>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mlpipeline.datasets import build_manifest, save_manifest, split_dataset, validate_dataset, write_yolo_split


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interim", required=True, help="interim root with flat images/ + labels/")
    ap.add_argument("--out", required=True, help="processed output root")
    ap.add_argument("--name", required=True)
    ap.add_argument("--classes", default="", help="comma-separated class names by id order")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group-sep", default="", help="stem prefix separator to group tiles (leakage guard)")
    ap.add_argument("--save-manifest", default="")
    args = ap.parse_args(argv)

    interim = Path(args.interim)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    img_dir = interim / "images"
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_paths = sorted(
        p.relative_to(img_dir).as_posix()
        for p in img_dir.rglob("*") if p.suffix.lower() in exts
    )
    if not image_paths:
        print(f"ERROR: no images under {img_dir}", file=sys.stderr)
        return 2

    # read classes per image for stratification
    image_classes: dict[str, list[str]] = {}
    for rel in image_paths:
        lbl = interim / "labels" / (Path(rel).stem + ".txt")
        names = []
        if lbl.is_file():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if parts and parts[0].isdigit():
                    cid = int(parts[0])
                    names.append(classes[cid] if cid < len(classes) else str(cid))
        image_classes[rel] = names

    fractions = {"train": args.train, "val": args.val, "test": args.test}
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        print(f"ERROR: fractions sum to {sum(fractions.values())}", file=sys.stderr)
        return 2

    assignment, meta = split_dataset(
        image_paths, image_classes, fractions, seed=args.seed,
        group_by_prefix_sep=args.group_sep or None,
    )
    processed = write_yolo_split(interim, Path(args.out), assignment)
    print(f"split written: {processed}")
    print(f"counts: {json.dumps(meta.counts)}")
    for s, dist in meta.per_split_classes.items():
        print(f"  {s}: {json.dumps(dist)}")

    manifest = build_manifest(
        processed, name=args.name, format="yolo", class_names=classes or None,
        split_seed=args.seed, notes=f"converted from {interim}; group_sep={args.group_sep or '-'}",
    )
    report = validate_dataset(manifest)
    print(f"validation: {'OK' if report.ok else 'FAILED'}")
    for issue in report.errors:
        print(f"  [error]   {issue.code}: {issue.detail}")
    for issue in report.warnings:
        print(f"  [warning] {issue.code}: {issue.detail}")

    if args.save_manifest:
        out = save_manifest(manifest, Path(args.save_manifest))
        print(f"manifest written: {out}")

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
