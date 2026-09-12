"""Detector protocol (Section 8.1 / ADR-002).

Implementations return RawDetection only (model-space boxes, bare class ids,
raw scores). They know NOTHING about class names, filtering, or geolocation.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from mlpipeline.config.schemas import DetectionConfig
from mlpipeline.datatypes.detection import RawDetection
from mlpipeline.datatypes.image import ProcessedSonarImage


class PredictParams:
    """Per-request threshold overrides (the ONLY overridable parameters).

    Bounds are enforced by the backend; the detector itself clamps defensively.
    """

    def __init__(
        self,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
        max_detections: int | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def resolve(self, defaults: DetectionConfig) -> DetectionConfig:
        return DetectionConfig(
            confidence_threshold=(
                defaults.confidence_threshold
                if self.confidence_threshold is None
                else min(max(self.confidence_threshold, 0.0), 1.0)
            ),
            iou_threshold=(
                defaults.iou_threshold
                if self.iou_threshold is None
                else min(max(self.iou_threshold, 0.0), 1.0)
            ),
            max_detections=defaults.max_detections
            if self.max_detections is None
            else max(1, self.max_detections),
        )


class ModelMeta:
    """What the API layer may know about a loaded model (no framework types)."""

    def __init__(
        self,
        model_version: str,
        architecture_family: str,
        class_map: dict[int, str],
        input_size: list[int],
        framework: str,
    ) -> None:
        self.model_version = model_version
        self.architecture_family = architecture_family
        self.class_map = class_map  # id -> class name (from registry, verbatim)
        self.input_size = input_size
        self.framework = framework


class DetectorLoadError(Exception):
    """Model weights missing/corrupt or incompatible — maps to MODEL_UNAVAILABLE."""


class DetectorPredictError(Exception):
    """Inference failure — maps to INFERENCE_FAILED."""


@runtime_checkable
class Detector(Protocol):
    """Uniform load → predict contract for every detector implementation."""

    def load(self, model_version: str) -> ModelMeta:
        """Load weights + metadata for the given registry version."""
        ...

    def predict(
        self,
        image: ProcessedSonarImage,
        pixels,
        params: PredictParams | None,
        defaults: DetectionConfig,
    ) -> list[RawDetection]:
        """Run inference. ``pixels`` is the processed numpy array matching
        ``image`` (kept separate to avoid shipping arrays through pydantic)."""
        ...

    def metadata(self) -> ModelMeta:
        """Metadata of the currently loaded model. Raises if not loaded."""
        ...

    @property
    def is_loaded(self) -> bool:
        ...
