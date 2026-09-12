#!/usr/bin/env python3
"""Site-disjointness audit for DRISHTI-SSS (shipwreck classes).

Derives site identifiers from filenames:
  - wreckA (AI4Shipwrecks): wreckA_<site>_y<int>_x<int>  -> site = <site>
  - wreckR (Roboflow):      wreckR_<obj>-<id>_png...     -> site = <obj>-<id>

Reports train/val/test site overlap and how many test tiles come from sites
also present in train/val (leakage), plus the size of a genuinely
site-disjoint test subset. Read-only; prints a JSON summary too.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "datasets" / "processed" / "drishti-sss"


def wreckA_site(stem: str) -> str | None:
    m = re.match(r"wreckA_(.+)_y\d+_x\d+$", stem)
    return m.group(1) if m else None


def wreckR_site(stem: str) -> str | None:
    m = re.match(r"wreckR_([a-z]+-\d+)_png", stem)
    return m.group(1) if m else None


def audit(pattern: str, extractor) -> dict:
    per_split = {}
    for split in ("train", "val", "test"):
        sites: Counter = Counter()
        for f in (ROOT / split / "images").glob(pattern):
            s = extractor(f.stem)
            if s:
                sites[s] += 1
        per_split[split] = sites

    tr, va, te = per_split["train"], per_split["val"], per_split["test"]
    ov_tr_te = set(tr) & set(te)
    ov_tr_va = set(tr) & set(va)
    ov_va_te = set(va) & set(te)
    leaked_tiles = sum(te[s] for s in ov_tr_te)
    te_total = sum(te.values())
    disjoint_sites = set(te) - set(tr) - set(va)
    disjoint_tiles = sum(te[s] for s in disjoint_sites)
    return {
        "train_sites": len(tr),
        "val_sites": len(va),
        "test_sites": len(te),
        "train_test_overlapping_sites": len(ov_tr_te),
        "train_val_overlapping_sites": len(ov_tr_va),
        "val_test_overlapping_sites": len(ov_va_te),
        "test_tiles_total": te_total,
        "test_tiles_from_sites_also_in_train_or_val": leaked_tiles,
        "test_tiles_leaked_pct": round(100 * leaked_tiles / max(te_total, 1), 1),
        "site_disjoint_test_sites": len(disjoint_sites),
        "site_disjoint_test_tiles": disjoint_tiles,
        "example_overlapping_sites": sorted(ov_tr_te)[:10],
        "example_disjoint_sites": sorted(disjoint_sites)[:10],
    }


def main() -> int:
    out = {
        "wreckA_AI4Shipwrecks": audit("wreckA*", wreckA_site),
        "wreckR_Roboflow": audit("wreckR*", wreckR_site),
    }
    for name, r in out.items():
        print(f"== {name}")
        for k, v in r.items():
            print(f"  {k}: {v}")
    print("\nJSON:")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
