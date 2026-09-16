#!/usr/bin/env python3
"""Operating-point sweep: how many shipwrecks do we surface, and at what cost?

The model's ranking quality (AP) is fixed, but the SERVING THRESHOLD decides
which candidates reach an analyst. This measures the precision/recall trade-off
per class across thresholds using the project's own evaluation pipeline
(`mlpipeline.evaluation.evaluate_model`) — the same code path that produced the
reported metrics — so nothing here is a parallel reimplementation.

Why this matters for this project: detections are not decisions. Every emitted
box survives to the UI carrying `filtering_status` (accepted/flagged/rejected)
and a reason, so a recall-first threshold trades review load for fewer missed
objects, and the deterministic filter is the precision backstop.

Tune on `val` (the selection split); report the chosen point on `test`.

    python scripts/sweep_operating_point.py --split val
    python scripts/sweep_operating_point.py --split val --tta
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlpipeline.evaluation.evaluate import evaluate_model
from mlpipeline.registry.eval_runs import EvalRunStore

ROOT = Path(__file__).resolve().parents[1]
MODEL = "drishti-ss_yolov8n_e30_final"
MANIFEST = ROOT / "datasets/manifests/drishti-sss.json"
# Sweep records are kept OUT of models/eval/<model>/ so they can never become the
# "latest" run that /models/{v}/metrics serves to the dashboard.
SWEEP_STORE = ROOT / "models/eval/sweep_runs"
OUT = ROOT / "models/eval/sweeps"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--thresholds", default="0.25,0.15,0.10",
                    help="comma-separated confidence thresholds")
    ap.add_argument("--tta", action="store_true", help="also run with TTA enabled")
    args = ap.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    store = EvalRunStore(SWEEP_STORE)
    rows: list[dict] = []

    print(f"model {MODEL} | split {args.split} | manifest {MANIFEST.name}\n")
    print(f"{'conf':>5} {'tta':>4} | {'shipwreck P':>11} {'R':>6} {'F1':>6} {'AP50':>6}"
          f" | {'ALL P':>6} {'R':>6} {'F1':>6} {'mAP50':>6}")
    for tta in ([False, True] if args.tta else [False]):
        for thr in thresholds:
            run = evaluate_model(
                MODEL,
                MANIFEST,
                split=args.split,
                registry=None,
                store=store,
                confidence_threshold=thr,
                tta=tta,
                notes=f"operating-point sweep (split={args.split}, conf={thr}, tta={tta})",
            )
            sw = run.per_class.get("shipwreck")
            m = run.metrics
            row = {
                "split": args.split,
                "confidence_threshold": thr,
                "tta": tta,
                "overall": {
                    "precision": m.precision, "recall": m.recall,
                    "f1": m.f1, "mAP50": m.mAP50,
                },
                "per_class": {
                    k: {"precision": v.precision, "recall": v.recall,
                        "f1": v.f1, "ap50": v.ap50, "support": v.support}
                    for k, v in run.per_class.items()
                },
                "eval_run_id": run.eval_run_id,
            }
            rows.append(row)
            print(
                f"{thr:>5.2f} {str(tta):>4} | "
                f"{(sw.precision if sw else float('nan')):>11.3f} "
                f"{(sw.recall if sw else float('nan')):>6.3f} "
                f"{(sw.f1 if sw else float('nan')):>6.3f} "
                f"{(sw.ap50 if sw else float('nan')):>6.3f} | "
                f"{(m.precision or 0):>6.3f} {(m.recall or 0):>6.3f} "
                f"{(m.f1 or 0):>6.3f} {(m.mAP50 or 0):>6.3f}"
            )

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"operating_point_{args.split}{'_tta' if args.tta else ''}.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    print(f"sweep eval records: {SWEEP_STORE.relative_to(ROOT)} (kept out of the servable store)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
