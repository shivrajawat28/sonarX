#!/usr/bin/env python3
"""Ultralytics val of the final model on the site-disjoint test subset.

Supplementary to the official test split: every tile here comes from a survey
sequence/site never seen in train or val (see scripts/audit_site_leakage.py).
Writes models/eval/e30_final_sitedisjoint_summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models/weights/drishti-ss_yolov8n_e30/weights/best_final.pt"
DATA = ROOT / "datasets/processed/drishti-sss-site-disjoint/data.yaml"
OUT = ROOT / "models/eval/e30_final_sitedisjoint_summary.json"


def main() -> int:
    model = YOLO(str(WEIGHTS))
    r = model.val(
        data=str(DATA), split="test", device="cpu", workers=2,
        plots=False, verbose=False,
        project=str(ROOT / "models/eval"),
        name="e30_final_sitedisjoint", exist_ok=True,
    )
    d = r.results_dict
    summary = {
        "model_version": "drishti-ss_yolov8n_e30_final",
        "split": "test-site-disjoint",
        "precision": round(d.get("metrics/precision(B)", float("nan")), 4),
        "recall": round(d.get("metrics/recall(B)", float("nan")), 4),
        "mAP50": round(d.get("metrics/mAP50(B)", float("nan")), 4),
        "mAP50-95": round(d.get("metrics/mAP50-95(B)", float("nan")), 4),
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
