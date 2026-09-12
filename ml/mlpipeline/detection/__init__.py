"""Detection subsystem: Detector protocol + adapters + model registry factory.

Contract (ADR-002): the app depends on `Detector` and canonical datatypes only.
Class names come from model metadata `class_map` — never from this package.
"""
from mlpipeline.detection.base import Detector, PredictParams, ModelMeta
from mlpipeline.detection.registry import (
    DetectorRegistry,
    register_detector,
    create_detector,
    known_detector_kinds,
)

# Import adapters so they self-register.
import mlpipeline.detection.stub as _stub  # noqa: F401
try:
    import mlpipeline.detection.yolo.adapter as _yolo  # noqa: F401
except ImportError:
    pass  # ultralytics not installed — YOLO adapter unavailable

__all__ = [
    "Detector",
    "PredictParams",
    "ModelMeta",
    "DetectorRegistry",
    "register_detector",
    "create_detector",
    "known_detector_kinds",
]
