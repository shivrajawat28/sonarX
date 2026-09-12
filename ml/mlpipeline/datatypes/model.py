"""Model lifecycle datatypes: ModelVersion, EvaluationRun, Report (Sections 6.4, 6.5, 13)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from mlpipeline.datatypes.common import new_id, now_utc_iso

ModelStatus = Literal["active", "shadow", "retired"]


class ConfigRef(BaseModel):
    """Reference to a versioned config artifact (path + content hash)."""

    path: str
    sha256: str


class ModelVersion(BaseModel):
    """Append-only registry entry: the authoritative answer to 'which classes,
    which preprocessing, which data, which hyperparameters' for a model."""

    model_version: str
    architecture_family: str  # e.g. "yolo" — final arch is OPEN decision #5
    framework: str = "pytorch"
    checkpoint_path: str
    input_size: list[int] = [640, 640]
    class_map: dict[str, str]  # e.g. 0 -> <class name from dataset>; never hardcoded
    preprocess_config_ref: ConfigRef
    train_dataset_ref: ConfigRef | None = None
    train_config: dict[str, Any] = Field(default_factory=dict)
    best_eval_ref: str | None = None  # eval_run_id
    status: ModelStatus = "shadow"
    notes: str = ""
    created_at: str = Field(default_factory=now_utc_iso)


class MetricsSummary(BaseModel):
    mAP50: float | None = None
    mAP50_95: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None


class PerClassMetrics(BaseModel):
    ap50: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    support: int | None = None


class EvaluationRun(BaseModel):
    """Immutable evaluation record — the ONLY source of metrics shown in the UI (ADR-010)."""

    eval_run_id: str = Field(default_factory=lambda: new_id("eval"))
    model_version: str
    dataset_ref: ConfigRef  # dataset manifest path + hash
    split: Literal["train", "val", "test"]
    split_seed: int
    preprocess_config_hash: str
    filter_config_hash: str | None = None  # metrics with filter ON vs OFF recorded separately
    filter_enabled: bool = False
    metrics: MetricsSummary = Field(default_factory=MetricsSummary)
    per_class: dict[str, PerClassMetrics] = Field(default_factory=dict)
    confusion_matrix_ref: str | None = None
    pr_curve_refs: list[str] = Field(default_factory=list)
    failure_case_refs: list[str] = Field(default_factory=list)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    timestamp: str = Field(default_factory=now_utc_iso)


class ReportStats(BaseModel):
    images: int = 0
    detections: int = 0
    accepted: int = 0
    flagged: int = 0
    rejected: int = 0


class Report(BaseModel):
    report_id: str = Field(default_factory=lambda: new_id("rpt"))
    kind: Literal["survey_summary", "run_summary"]
    subject_ref: str  # survey_id or run_id
    format: Literal["pdf"] = "pdf"
    artifact_path: str | None = None
    generated_at: str | None = None
    model_version: str | None = None
    content_stats: ReportStats = Field(default_factory=ReportStats)
