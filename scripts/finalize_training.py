#!/usr/bin/env python3
"""Finalize the 30-epoch training run once it completes.

Idempotent. Safe to run repeatedly. Does:
  1. (optionally --wait) poll until the training process exits
  2. register `drishti-ss_yolov8n_e30_final` from the run's best.pt
  3. evaluate it on the TEST split via ml.scripts.evaluate (stored EvaluationRun)
  4. ultralytics val on val + test splits (mAP50 / mAP50-95 / P / R)
  5. registry statuses: interim -> retired, final -> active

Usage:
    python scripts/finalize_training.py            # run once, fail if training still active
    python scripts/finalize_training.py --wait     # poll (up to --timeout-hours) until done
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "ml"))

RUN_DIR = project_root / "models" / "weights" / "drishti-ss_yolov8n_e30"
BEST = RUN_DIR / "weights" / "best.pt"
LAST = RUN_DIR / "weights" / "last.pt"
FINAL_VERSION = "drishti-ss_yolov8n_e30_final"
INTERIM_VERSION = "drishti-ss_yolov8n_e5_interim"
MANIFEST = project_root / "datasets" / "manifests" / "drishti-sss.json"
DATA_YAML = project_root / "datasets" / "processed" / "drishti-sss" / "data.yaml"


def trainer_running() -> bool:
    lock = project_root / "scripts" / ".training.lock"
    if not lock.is_file():
        return False
    raw = lock.read_text().strip()
    if not raw.isdigit():
        return False
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {raw}"], capture_output=True, text=True, timeout=15
    ).stdout
    return f" {raw} " in out or out.rstrip().endswith(raw)


def train_epochs_done() -> int:
    """Epochs recorded in results.csv (1 row per epoch after de-dup)."""
    csv = RUN_DIR / "results.csv"
    if not csv.is_file():
        return 0
    lines = [ln for ln in csv.read_text().splitlines()[1:] if ln.strip()]
    return len(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wait", action="store_true", help="poll until the trainer exits")
    ap.add_argument("--timeout-hours", type=float, default=14.0)
    ap.add_argument("--allow-partial", action="store_true",
                    help="proceed even if <30 epochs recorded (trainer may have crashed); "
                         "the registry notes will state the true epoch count")
    args = ap.parse_args()

    if args.wait:
        print(f"waiting for training to finish (up to {args.timeout_hours} h)...")
        deadline = time.time() + args.timeout_hours * 3600
        while time.time() < deadline:
            if not trainer_running():
                break
            time.sleep(300)
        else:
            print("ERROR: timed out waiting for trainer")
            return 1
        time.sleep(30)  # let ultralytics finish writing

    if trainer_running():
        print("REFUSED: training is still running. Use --wait or run later.")
        return 2

    epochs = train_epochs_done()
    print(f"training complete; epochs recorded in results.csv: {epochs}")
    if epochs < 30 and not args.allow_partial:
        print(
            f"REFUSED: only {epochs}/30 epochs recorded — the trainer likely crashed "
            f"before completion. Refusing to label a partial run as '{FINAL_VERSION}'."
        )
        print("Re-run with --allow-partial to finalize anyway (true epoch count is "
              "recorded in the registry notes), or resume training first.")
        return 3
    if not BEST.is_file():
        print(f"ERROR: {BEST} missing")
        return 1

    # 2. register final model (idempotent: skip if already registered)
    #    FREEZE the weights first: best.pt keeps being overwritten during
    #    training/resume, and a registry entry must point at an immutable
    #    checkpoint (bugfix: the interim entry pointed at the live best.pt,
    #    so 'e5_interim' silently became ~e11 weights).
    from mlpipeline.registry.models import get_registry

    reg = get_registry()
    frozen = RUN_DIR / "weights" / "best_final.pt"
    if not frozen.is_file():
        import shutil

        src = BEST if BEST.is_file() else LAST
        shutil.copy2(src, frozen)
        print(f"frozen weights: {frozen} (copy of {src.name})")
    if reg.get(FINAL_VERSION) is None:
        cmd = [
            sys.executable, str(project_root / "scripts" / "register_trained_model.py"),
            "--version", FINAL_VERSION,
            "--checkpoint", str(frozen.relative_to(project_root)),
            "--status", "shadow",
            "--notes",
            (f"YOLOv8n, target 30 epochs (single-instance resumed run) on "
             f"DRISHTI-SSS train split; {epochs}/30 epochs recorded; CPU. "
             f"{'COMPLETE run.' if epochs >= 30 else 'PARTIAL run (trainer ended early) — epoch count is honest in this note.'}"),
        ]
        print("registering:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=project_root)
    else:
        print(f"{FINAL_VERSION} already registered")

    # 3. repo eval on TEST split (stored immutable EvaluationRun)
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root / 'ml'};{project_root}"
    evals = subprocess.run(
        [sys.executable, "-m", "ml.scripts.evaluate",
         "--model", FINAL_VERSION,
         "--manifest", str(MANIFEST),
         "--split", "test",
         "--notes", "Final 30-epoch model, test split, filter OFF (detector-only)."],
        capture_output=True, text=True, cwd=project_root, env=env,
    )
    print(evals.stdout[-1500:] or evals.stderr[-1500:])
    if evals.returncode != 0:
        print("ERROR: test-split evaluation failed (see above)")
        return 1

    # 4. ultralytics val on val + test (mAP family) — on the FROZEN weights
    from ultralytics import YOLO

    model = YOLO(str(frozen))
    for split in ("val", "test"):
        r = model.val(
            data=str(DATA_YAML), split=split, device="cpu", workers=2,
            project=str(project_root / "models" / "eval"),
            name=f"e30_final_{split}", exist_ok=True, plots=True, verbose=False,
        )
        d = r.results_dict
        summary = {
            "model_version": FINAL_VERSION,
            "split": split,
            "precision": round(d.get("metrics/precision(B)", float("nan")), 4),
            "recall": round(d.get("metrics/recall(B)", float("nan")), 4),
            "mAP50": round(d.get("metrics/mAP50(B)", float("nan")), 4),
            "mAP50-95": round(d.get("metrics/mAP50-95(B)", float("nan")), 4),
        }
        out = project_root / "models" / "eval" / f"e30_final_{split}_summary.json"
        out.write_text(json.dumps(summary, indent=2))
        print(split, summary)

    # 5. status promotion
    reg.invalidate()
    reg.set_status(INTERIM_VERSION, "retired")
    reg.set_status(FINAL_VERSION, "active")
    print(f"registry: {INTERIM_VERSION} -> retired; {FINAL_VERSION} -> active")
    print("DONE. Post-steps:")
    print("  1. RESTART the backend (a running server still holds the old model in memory)")
    print("  2. Update docs/EVALUATION.md + docs/ENGINEERING_REPORT.md with the new numbers")
    print("  3. Re-run the browser E2E (frontend/e2e_full.mjs) against the final model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
