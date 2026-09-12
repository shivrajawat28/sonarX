"""CLI: train a model and register it (Step 7 gate: tiny train → eval → register).

The registry is append-only: `--version` must be a NEW id. Classes come from the
dataset config (data evidence), never hardcoded here.

    # Smoke/test path (no torch needed):
    PYTHONPATH=ml:. python -m ml.scripts.train \
        --version stub-exp-001 --dataset-config ml/configs/datasets/my_ds.yaml --backend stub

    # Real training (requires torch + ultralytics and a real converted dataset):
    PYTHONPATH=ml:. python -m ml.scripts.train \
        --version yolo-exp-001 --dataset-config ml/configs/datasets/my_ds.yaml --backend yolo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mlpipeline.training.train import train_model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", required=True,
                    help="new model version id (registry is append-only; duplicates are rejected)")
    ap.add_argument("--dataset-config", required=True,
                    help="dataset config YAML; candidate_classes define the class map")
    ap.add_argument("--preprocess-config",
                    default=str(Path(__file__).resolve().parents[1] / "configs/preprocessing/baseline_sonar.yaml"))
    ap.add_argument("--backend", default="stub", choices=["stub", "yolo"],
                    help="stub = smoke/test fixture only; yolo = real training (torch + ultralytics)")
    ap.add_argument("--train-config", default="",
                    help="optional JSON file with train hyperparameters (e.g. {\"input_size\": [640, 640]})")
    args = ap.parse_args(argv)

    train_config: dict = {}
    if args.train_config:
        train_config = json.loads(Path(args.train_config).read_text())

    try:
        entry = train_model(
            args.version,
            args.dataset_config,
            args.preprocess_config,
            backend=args.backend,
            train_config=train_config,
        )
    except Exception as e:  # noqa: BLE001 — CLI boundary: print, don't traceback
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"registered '{entry.model_version}' status={entry.status} framework={entry.framework}")
    print(f"class_map: {entry.class_map}")
    if args.backend == "stub":
        print("NOTE: stub backend produces a labelled TEST/DEMO checkpoint — it is not a trained detector.")
    print("next: evaluate with `make evaluate` / ml.scripts.evaluate, then promote status in the registry if merited")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
