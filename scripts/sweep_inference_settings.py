#!/usr/bin/env python3
"""Inference-setting sweep for recall (no retraining).

Searches inference-time settings that can lift the weakest real class
(shipwreck) without paying for another CPU training run:

  * input resolution  — the model is fully convolutional, so evaluating above
    the 640px training size gives small/partial objects more pixels.
  * TTA (`augment=True`) — averaged multi-scale/flip inference.
  * confidence threshold — the operating point, i.e. how many candidates are
    surfaced for review before deterministic filtering prunes them.

METHODOLOGY: settings are compared on the **val** split (the model-selection
split, per docs/EVALUATION.md). The chosen setting is then reported on the
**test** split only once — test is never used to choose a setting.

    python scripts/sweep_inference_settings.py --split val
    python scripts/sweep_inference_settings.py --split test --final
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models/weights/drishti-ss_yolov8n_e30/weights/best_final.pt"
DATA = ROOT / "datasets/processed/drishti-sss/data.yaml"
OUT_DIR = ROOT / "models/eval/sweeps"

# shipwreck is the class we are trying to improve; the rest are guard rails so a
# change that lifts shipwreck by wrecking the others is visible immediately.
WATCH = ("shipwreck", "mine_cylinder", "submarine_pipeline", "ghost_net")


def run(model: YOLO, *, imgsz: int, augment: bool, split: str, name: str) -> dict:
    r = model.val(
        data=str(DATA),
        split=split,
        imgsz=imgsz,
        augment=augment,
        device="cpu",
        workers=2,
        plots=False,
        verbose=False,
        project=str(OUT_DIR),
        name=name,
        exist_ok=True,
    )
    d = r.results_dict
    # Per-class rows are indexed by the dataset's class names.
    names = r.names if isinstance(r.names, dict) else dict(enumerate(r.names))
    per_class = {}
    try:
        for i, cname in names.items():
            per_class[cname] = {
                "precision": float(r.box.p[i]),
                "recall": float(r.box.r[i]),
                "mAP50": float(r.box.ap50[i]),
                "mAP50-95": float(r.box.ap[i]),
            }
    except (AttributeError, IndexError, TypeError) as e:  # pragma: no cover
        print(f"  (per-class unavailable: {e})")

    row = {
        "setting": name,
        "split": split,
        "imgsz": imgsz,
        "tta": augment,
        "precision": float(d.get("metrics/precision(B)", float("nan"))),
        "recall": float(d.get("metrics/recall(B)", float("nan"))),
        "mAP50": float(d.get("metrics/mAP50(B)", float("nan"))),
        "mAP50-95": float(d.get("metrics/mAP50-95(B)", float("nan"))),
        "per_class": per_class,
    }
    sw = per_class.get("shipwreck", {})
    print(
        f"  {name:22s} P {row['precision']:.3f}  R {row['recall']:.3f}  "
        f"mAP50 {row['mAP50']:.4f}  mAP50-95 {row['mAP50-95']:.4f}"
        f"  | shipwreck R {sw.get('recall', float('nan')):.3f} "
        f"AP50 {sw.get('mAP50', float('nan')):.3f}"
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true", help="run only the chosen setting")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    if not WEIGHTS.is_file():
        print(f"ERROR: frozen weights missing: {WEIGHTS}")
        return 1

    model = YOLO(str(WEIGHTS))
    rows: list[dict] = []
    print(f"weights: {WEIGHTS.relative_to(ROOT)}  split: {args.split}\n")

    if args.final:
        rows.append(run(model, imgsz=args.imgsz, augment=args.tta, split=args.split,
                        name=f"final_{args.split}_{args.imgsz}{'_tta' if args.tta else ''}"))
    else:
        for imgsz in (640, 960, 1280):
            rows.append(run(model, imgsz=imgsz, augment=False, split=args.split,
                            name=f"sweep_{args.split}_{imgsz}"))
        # TTA is ~3x the cost, so only probe it at the most promising resolution
        best = max(rows, key=lambda r: r["per_class"].get("shipwreck", {}).get("mAP50", 0))
        rows.append(run(model, imgsz=best["imgsz"], augment=True, split=args.split,
                        name=f"sweep_{args.split}_{best['imgsz']}_tta"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"inference_sweep_{args.split}.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
