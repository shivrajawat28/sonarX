"""Filter rule protocol + shared context (Section 9.2)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mlpipeline.datatypes.detection import Detection


@dataclass
class Verdict:
    """One rule's outcome: a penalty in [0, 1] plus an explainable reason."""

    rule: str
    penalty: float  # 0.0 = no penalty
    reason: str = ""  # human-readable, recorded in filter_reasons
    triggered: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.penalty <= 1.0:
            raise ValueError(f"penalty must be in [0,1], got {self.penalty}")


@dataclass
class FilterContext:
    """Everything a rule may need about the image the detections came from."""

    image_width: int
    image_height: int
    pixels: np.ndarray | None = None  # source-space grayscale [0,1] when available
    image_intensity_p10: float | None = None
    image_intensity_p90: float | None = None
    class_map: dict[int, str] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class FilterRule(Protocol):
    """Every rule implements: name + evaluate(detection, ctx, params) -> Verdict."""

    name: str

    def evaluate(self, detection: Detection, ctx: FilterContext, params: dict[str, Any]) -> Verdict: ...
