#!/usr/bin/env python3
"""Train YOLOv8n on DRISHTI-SSS dataset.

Usage:
    python scripts/train_yolo.py [--epochs 50] [--batch 8] [--imgsz 640]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8n on DRISHTI-SSS")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs")
    p.add_argument("--batch", type=int, default=8, help="Batch size")
    p.add_argument("--imgsz", type=int, default=640, help="Image size")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Base model")
    p.add_argument("--name", type=str, default=None, help="Run name")
    p.add_argument("--resume", type=str, default="", help="Path to last.pt: resume an interrupted run (all other hyperparams come from the checkpoint)")
    return p.parse_args()


def main():
    args = parse_args()
    
    from ultralytics import YOLO
    
    # Paths
    project_root = Path(__file__).resolve().parents[1]
    data_yaml = project_root / "datasets" / "processed" / "drishti-sss" / "data.yaml"
    models_root = project_root / "models" / "weights"

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise SystemExit(f"Resume checkpoint not found: {resume_path}")
        print(f"=== RESUMING interrupted run from {resume_path} ===")
        print(f"Start time: {datetime.now(timezone.utc).isoformat()}")
        model = YOLO(str(resume_path))
        # Resume replays ALL original args (data, epochs, batch, seed, aug) from
        # the checkpoint — passing them again risks conflicts. resume=True only.
        model.train(resume=True)
        print(f"\n=== Resumed run complete: {datetime.now(timezone.utc).isoformat()} ===")
        return None, resume_path.parent / "best.pt"
    
    if not data_yaml.exists():
        raise SystemExit(f"Data YAML not found: {data_yaml}")
    
    # Load data config to get class info
    with open(data_yaml, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    
    run_name = args.name or f"drishti-ss_{args.model.replace('.pt','')}_e{args.epochs}_b{args.batch}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    weights_dir = models_root / run_name
    
    print(f"=== Training Configuration ===")
    print(f"  Model:      {args.model}")
    print(f"  Dataset:    {data_yaml}")
    print(f"  Classes:    {data_cfg.get('nc', '?')} — {data_cfg.get('names', {})}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch:      {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Seed:       {args.seed}")
    print(f"  Device:     cpu")
    print(f"  Output:     {weights_dir}")
    print(f"  Start time: {datetime.now(timezone.utc).isoformat()}")
    print()
    
    t0 = time.time()
    
    # Train
    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        seed=args.seed,
        device="cpu",
        project=str(models_root),
        name=run_name,
        exist_ok=True,
        verbose=True,
        patience=20,  # early stopping patience
        save=True,
        save_period=10,  # save every 10 epochs
        plots=True,
        # Sonar-safe augmentations (grayscale sonar imagery)
        augment=True,
        hsv_h=0.0,    # NO hue shift — sonar is grayscale
        hsv_s=0.0,    # NO saturation — grayscale
        hsv_v=0.3,    # brightness variation only
        degrees=5.0,  # small rotation
        translate=0.1, # small translation
        scale=0.3,    # moderate scale
        shear=2.0,    # small shear
        perspective=0.0,  # no perspective
        flipud=0.0,   # vertical flip: DEBATABLE for sonar, keep off
        fliplr=0.5,   # horizontal flip: generally safe for sonar
        mosaic=0.8,   # mosaic augmentation
        mixup=0.1,    # light mixup
        erasing=0.2,  # random erasing
    )
    
    elapsed = time.time() - t0
    print(f"\n=== Training Complete ===")
    print(f"  Duration: {elapsed/60:.1f} minutes")
    
    # Find best weights
    best_path = weights_dir / "train" / "weights" / "best.pt"
    last_path = weights_dir / "train" / "weights" / "last.pt"
    
    # Ultralytics puts weights under project/name/weights/
    alt_best = models_root / run_name / "weights" / "best.pt"
    if best_path.exists():
        print(f"  Best weights: {best_path}")
    elif alt_best.exists():
        best_path = alt_best
        print(f"  Best weights: {best_path}")
    else:
        # Search for best.pt
        for p in weights_dir.rglob("best.pt"):
            best_path = p
            print(f"  Best weights: {best_path}")
            break
    
    # Save training metadata
    metadata = {
        "run_name": run_name,
        "model_architecture": args.model,
        "dataset": str(data_yaml),
        "dataset_name": "drishti-sss",
        "classes": data_cfg.get("names", {}),
        "nc": data_cfg.get("nc", 4),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "device": "cpu",
        "preprocessing": "already_preprocessed (Lee + CLAHE applied upstream)",
        "augmentation": "ultralytics defaults + sonar-safe overrides",
        "base_weights": args.model,
        "training_duration_sec": elapsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "best_weights": str(best_path),
        "notes": "YOLOv8n trained on DRISHTI-SSS (CC-BY-SA-4.0). "
                 "Data already preprocessed with Lee speckle + CLAHE. "
                 "ghost_net class is 100% synthetic.",
    }
    
    meta_path = weights_dir / "train" / "training_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  Metadata: {meta_path}")
    
    # Also copy best.pt to a stable location
    stable_best = weights_dir / "best.pt"
    if best_path.exists() and not stable_best.exists():
        shutil.copy2(best_path, stable_best)
        print(f"  Stable copy: {stable_best}")
    
    print(f"\n=== Next steps ===")
    print(f"  1. Evaluate: python -m ml.scripts.evaluate --model {run_name}")
    print(f"  2. Or use ultralytics val: yolo detect val model={best_path} data={data_yaml}")
    
    return run_name, best_path


if __name__ == "__main__":
    main()
