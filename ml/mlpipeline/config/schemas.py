"""Pydantic config models for every configurable subsystem.

Rules (architecture Section 14):
- No thresholds, class names, model paths, or image sizes hardcoded in code.
- Configs are content-hashed; the hash travels with model versions and results.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class PreprocessOpSpec(BaseModel):
    """One op in the preprocessing chain: registry name + its parameters."""

    op: str
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class PreprocessConfig(BaseModel):
    """Ordered preprocessing chain (Section 7). Order = list order."""

    name: str = "baseline_sonar"
    description: str = ""
    ops: list[PreprocessOpSpec] = Field(min_length=1)

    @field_validator("ops")
    @classmethod
    def _unique_ops(cls, v: list[PreprocessOpSpec]) -> list[PreprocessOpSpec]:
        names = [o.op for o in v]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate preprocessing ops not allowed: {sorted(dupes)}")
        return v


class DetectionConfig(BaseModel):
    """Serving defaults for the detector + post-processing (Section 8).

    These are defaults; per-request overrides are allowed for thresholds only
    and bounded by backend settings (never below MIN_CONFIDENCE_OVERRIDE).
    """

    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=300, ge=1)
    # Class-agnostic: classes come from model class_map at inference time.


class FilterRuleSpec(BaseModel):
    """One false-positive filter rule: registry name + params + optional class scoping.

    ``applies_to`` empty = applies to all classes; otherwise list of class names
    (resolved at runtime from the model class_map — never hardcoded).
    """

    rule: str
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    applies_to: list[str] = Field(default_factory=list)
    penalty: float = Field(default=0.0, ge=0.0, le=1.0)  # default penalty cap for this rule


class FilterPolicy(BaseModel):
    """final_confidence -> filtering_status mapping (Section 9.4)."""

    accept_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    flag_threshold: float = Field(default=0.35, ge=0.0, le=1.0)

    @field_validator("flag_threshold")
    @classmethod
    def _flag_below_accept(cls, v: float, info) -> float:
        accept = info.data.get("accept_threshold")
        if accept is not None and v > accept:
            raise ValueError("flag_threshold must be <= accept_threshold")
        return v


class FilterConfig(BaseModel):
    name: str = "rules_baseline"
    description: str = ""
    enabled_rules: list[FilterRuleSpec] = Field(default_factory=list)
    policy: FilterPolicy = Field(default_factory=FilterPolicy)


class DatasetConfig(BaseModel):
    """Dataset definition (Section 5 / Phase 3): paths + format + candidate classes.

    OPEN decision #1 (final classes): this config is where a class list is
    declared once evidence exists; nothing else in the app may hardcode classes.
    """

    name: str
    root: str  # dataset root directory
    format: str  # yolo | coco | voc | csv — converter registry keys (Phase 3)
    candidate_classes: list[str] = Field(default_factory=list)  # verbatim from data review
    split_seed: int = 42
    split_fractions: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.7, "val": 0.2, "test": 0.1}
    )
    notes: str = ""

    @field_validator("split_fractions")
    @classmethod
    def _fractions_sum(cls, v: dict[str, float]) -> dict[str, float]:
        if abs(sum(v.values()) - 1.0) > 1e-6:
            raise ValueError(f"split_fractions must sum to 1.0, got {sum(v.values())}")
        return v
