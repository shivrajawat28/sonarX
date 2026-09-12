"""Training wrapper (Section 8.2 / ADR-009).

Two backends:
- "yolo": ultralytics-based training (requires the ml extras; OPEN decision #5).
- "stub": a deterministic non-learning 'training' that records a valid registry
  entry pointing at a marker file. Used ONLY by the smoke test to verify the
  train -> eval -> register chain (explicitly NOT a real model; status=shadow
  and notes make this unambiguous).

The wrapper, not the framework, is imported everywhere.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from mlpipeline.config.loader import load_config_file
from mlpipeline.config.schemas import DatasetConfig, PreprocessConfig
from mlpipeline.datatypes.model import ModelVersion
from mlpipeline.registry.models import ModelRegistry, get_registry, repo_root
from mlpipeline.training.experiment import ExperimentRecord, save_experiment


class TrainingError(Exception):
    pass


def train_model(
    model_version: str,
    dataset_config_path: str | Path,
    preprocess_config_path: str | Path,
    backend: str = "yolo",
    train_config: dict[str, Any] | None = None,
    registry: ModelRegistry | None = None,
    models_root: str | Path | None = None,
) -> ModelVersion:
    """Train (or smoke-train) a model and append it to the registry."""
    registry = registry or get_registry()
    if registry.get(model_version) is not None:
        raise TrainingError(
            f"model version '{model_version}' already registered — use a new version id"
        )

    ds_cfg, _ = load_config_file(dataset_config_path, DatasetConfig)
    pp_cfg, pp_hash = load_config_file(preprocess_config_path, PreprocessConfig)

    models_root = Path(models_root) if models_root else Path(__file__).resolve().parents[3] / "models"
    weights_dir = models_root / "weights" / model_version

    if backend == "yolo":
        _train_yolo(ds_cfg, pp_cfg, weights_dir, train_config or {})
        checkpoint = weights_dir / "best.pt"
        framework = "pytorch"
    elif backend == "stub":
        checkpoint = _train_stub(weights_dir)
        framework = "numpy"
    else:
        raise TrainingError(f"unknown training backend '{backend}'")

    record = ExperimentRecord(
        model_version=model_version,
        dataset_manifest=str(ds_cfg.root),
        dataset_manifest_sha256="",
        preprocess_config_path=str(preprocess_config_path),
        preprocess_config_sha256=pp_hash,
        train_config=train_config or {},
        notes=f"backend={backend}",
    )
    save_experiment(record, weights_dir)

    entry = ModelVersion(
        model_version=model_version,
        architecture_family=backend,
        framework=framework,
        checkpoint_path=_repo_relative(checkpoint),
        input_size=list((train_config or {}).get("input_size", [640, 640])),
        # Classes come from the dataset config (data evidence), never invented here.
        class_map={str(i): c for i, c in enumerate(ds_cfg.candidate_classes)},
        preprocess_config_ref={"path": _repo_relative(preprocess_config_path), "sha256": pp_hash},
        train_config=train_config or {},
        status="shadow",  # promoted to active only after evaluation
        notes=f"backend={backend}; classes from {Path(dataset_config_path).name}",
    )
    return registry.register(entry)


def _repo_relative(path: str | Path) -> str:
    """Store repo-relative paths so registry entries survive a repo move."""
    p = Path(path)
    try:
        return p.resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        return p.as_posix()


def _train_stub(weights_dir: Path) -> Path:
    """Marker checkpoint for the smoke chain. Clearly labeled; predicts nothing."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    marker = weights_dir / "stub_checkpoint.json"
    marker.write_text(
        '{"kind": "stub_checkpoint", "notice": "TEST/DEMO fixture — not a trained model"}',
        encoding="utf-8",
    )
    return marker


def _train_yolo(ds_cfg: DatasetConfig, pp_cfg: PreprocessConfig, weights_dir: Path, cfg: dict) -> None:
    """Real YOLO training via ultralytics. Requires torch + ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise TrainingError(
            f"YOLO training requires the ml extras (torch, ultralytics): {e}"
        ) from e

    from mlpipeline.training.augmentation import FORBIDDEN_AUGS

    chosen = set(cfg.get("augmentation", []))
    banned = chosen & set(FORBIDDEN_AUGS)
    if banned:
        raise TrainingError(f"sonar-unsafe augmentations refused: {sorted(banned)}")

    data_yaml = _write_yolo_data_yaml(ds_cfg, weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(cfg.get("base_weights", "yolo11n.pt"))
    model.train(
        data=str(data_yaml),
        epochs=int(cfg.get("epochs", 100)),
        batch=int(cfg.get("batch", 16)),
        imgsz=int(cfg.get("imgsz", 640)),
        seed=int(cfg.get("seed", 42)),
        project=str(weights_dir.parent),
        name=weights_dir.name,
        exist_ok=True,
        verbose=False,
    )
    best = weights_dir / "train" / "weights" / "best.pt"
    if not best.is_file():
        raise TrainingError(f"training finished but best.pt missing under {weights_dir}")


def _write_yolo_data_yaml(ds_cfg: DatasetConfig, weights_dir: Path) -> Path:
    import yaml

    root = Path(ds_cfg.root).resolve()
    payload = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: c for i, c in enumerate(ds_cfg.candidate_classes)},
    }
    weights_dir.mkdir(parents=True, exist_ok=True)
    out = weights_dir / "data.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out
