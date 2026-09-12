"""Result datatypes: what the inference engine produces and what gets persisted."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from mlpipeline.datatypes.common import new_id, now_utc_iso
from mlpipeline.datatypes.detection import Detection


class StageTimings(BaseModel):
    """Per-stage milliseconds — observability contract (Section 16)."""

    read_ms: float = 0.0
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    filter_ms: float = 0.0
    geolocate_ms: float = 0.0


class InferenceResult(BaseModel):
    """Canonical end-to-end result of running the pipeline on one image."""

    image_id: str
    model_version: str
    preprocess_config_hash: str
    preprocess_config_name: str | None = None
    filter_config_hash: str | None = None
    overrides_applied: dict = Field(default_factory=dict)
    detections: list[Detection] = Field(default_factory=list)
    timings_ms: StageTimings = Field(default_factory=StageTimings)
    warnings: list[str] = Field(default_factory=list)
    processed_image_ref: str | None = None  # artifact path of preprocessed image
    created_at: str = Field(default_factory=now_utc_iso)


class DetectionRun(BaseModel):
    """Persisted record of an inference run (single image or survey batch)."""

    run_id: str = Field(default_factory=lambda: new_id("run"))
    kind: Literal["single_image", "survey_batch"]
    image_id: str | None = None
    survey_id: str | None = None
    model_version: str
    preprocess_config_hash: str
    filter_config_hash: str | None = None
    overrides_applied: dict = Field(default_factory=dict)
    detection_ids: list[str] = Field(default_factory=list)
    image_ids: list[str] = Field(default_factory=list)  # for survey batches
    timings_ms: StageTimings = Field(default_factory=StageTimings)
    warnings: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_utc_iso)
