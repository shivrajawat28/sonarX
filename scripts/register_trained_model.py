#!/usr/bin/env python3
"""Register a real trained YOLO checkpoint in the model registry (append-only).

Reads class names from the dataset data.yaml (source of truth — never
hardcoded), hashes the preprocessing config into the registry record, and
appends the entry. Refuses to overwrite existing version ids.

Usage:
    python scripts/register_trained_model.py \
        --version drishti-ss_yolov8n_e30_final \
        --checkpoint models/weights/drishti-ss_yolov8n_e30/weights/best.pt \
        --status active \
        --notes "YOLOv8n 30 epochs on DRISHTI-SSS (CPU)"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "ml"))

from mlpipeline.datatypes.model import ModelVersion  # noqa: E402
from mlpipeline.registry.models import get_registry  # noqa: E402


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def repo_relative(path: Path) -> str:
    """Store repo-relative POSIX paths so the registry survives a repo move.

    An absolute path baked into registry.json breaks the moment the checkout is
    moved or synced, which silently degrades serving to MODEL_UNAVAILABLE.
    """
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()  # genuinely outside the repo — keep it absolute


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", required=True, help="new unique model version id")
    ap.add_argument("--checkpoint", required=True, help="path to best.pt / last.pt")
    ap.add_argument("--data-yaml", default=str(project_root / "datasets/processed/drishti-sss/data.yaml"))
    ap.add_argument("--preprocess-config", default=str(project_root / "ml/configs/preprocessing/drishti_preprocessed.yaml"))
    ap.add_argument("--status", default="shadow", choices=["shadow", "active", "retired"])
    ap.add_argument("--notes", default="")
    ap.add_argument("--train-config-json", default="", help="optional JSON file with train hyperparameters")
    ap.add_argument("--eval-run-id", default=None, help="link an EvaluationRun id if one exists")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"ERROR: checkpoint not found: {ckpt}")
        return 1
    if not ckpt.is_absolute():
        ckpt = project_root / ckpt

    data_yaml = Path(args.data_yaml)
    with open(data_yaml, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    names = data_cfg.get("names") or {}
    class_map = {str(int(k)): v for k, v in names.items()} if isinstance(names, dict) else {
        str(i): n for i, n in enumerate(names)
    }
    if not class_map:
        print(f"ERROR: no class names in {data_yaml}")
        return 1

    pp = Path(args.preprocess_config)
    if not pp.is_absolute():
        pp = project_root / pp

    train_config: dict = {}
    if args.train_config_json:
        train_config = json.loads(Path(args.train_config_json).read_text(encoding="utf-8"))
    train_config.setdefault("dataset", repo_relative(data_yaml))

    entry = ModelVersion(
        model_version=args.version,
        architecture_family="yolo",
        framework="pytorch",
        checkpoint_path=repo_relative(ckpt),
        input_size=[640, 640],
        class_map=class_map,
        preprocess_config_ref={"path": repo_relative(pp), "sha256": sha256_file(pp)},
        train_dataset_ref={"path": repo_relative(data_yaml), "sha256": sha256_file(data_yaml)},
        train_config=train_config,
        best_eval_ref=args.eval_run_id,
        status=args.status,
        notes=args.notes,
    )

    registry = get_registry()
    try:
        registry.register(entry)
    except Exception as e:  # noqa: BLE001 — CLI boundary
        print(f"ERROR: {e}")
        return 1
    print(f"registered '{entry.model_version}' status={entry.status} classes={class_map}")
    print(f"checkpoint: {entry.checkpoint_path}")
    print("NOTE: dataset/weights are referenced relative to the repo root.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
