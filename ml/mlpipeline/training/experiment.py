"""Experiment capture: a training run's full config snapshot (Section 6.2).

Saved beside the weights so any checkpoint can be traced to its data/config.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from mlpipeline.datatypes.common import now_utc_iso


class ExperimentRecord(BaseModel):
    model_version: str
    started_at: str = Field(default_factory=now_utc_iso)
    dataset_manifest: str
    dataset_manifest_sha256: str
    preprocess_config_path: str
    preprocess_config_sha256: str
    train_config: dict = Field(default_factory=dict)
    notes: str = ""


def save_experiment(record: ExperimentRecord, weights_dir: Path) -> Path:
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    out = weights_dir / "experiment.json"
    out.write_text(json.dumps(record.model_dump(), indent=2), encoding="utf-8")
    return out


def load_experiment(weights_dir: Path) -> ExperimentRecord | None:
    p = Path(weights_dir) / "experiment.json"
    if not p.is_file():
        return None
    return ExperimentRecord.model_validate(json.loads(p.read_text(encoding="utf-8")))
