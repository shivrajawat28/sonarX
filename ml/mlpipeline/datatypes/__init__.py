"""Canonical datatypes: the single source of truth shared by every pipeline stage.

API schemas (backend/app/schemas) mirror these but never leak mlpipeline classes.
"""
from mlpipeline.datatypes.image import SonarImage, ImageRecord, ProcessedSonarImage, ScaleFactors
from mlpipeline.datatypes.detection import (
    BBox,
    RawDetection,
    Detection,
    FilterStatus,
    GeoStatus,
    GeoProvenance,
)
from mlpipeline.datatypes.navigation import NavigationSample, NavigationTrack
from mlpipeline.datatypes.survey import Survey, SurveyManifest
from mlpipeline.datatypes.results import InferenceResult, DetectionRun, StageTimings
from mlpipeline.datatypes.model import ModelVersion, EvaluationRun, Report

__all__ = [
    "SonarImage",
    "ImageRecord",
    "ProcessedSonarImage",
    "ScaleFactors",
    "BBox",
    "RawDetection",
    "Detection",
    "FilterStatus",
    "GeoStatus",
    "GeoProvenance",
    "NavigationSample",
    "NavigationTrack",
    "Survey",
    "SurveyManifest",
    "InferenceResult",
    "DetectionRun",
    "StageTimings",
    "ModelVersion",
    "EvaluationRun",
    "Report",
]
