"""Confidence threshold application — recorded, never hidden (Section 6.3)."""
from __future__ import annotations

from mlpipeline.datatypes.detection import RawDetection


def apply_confidence_threshold(
    raw: list[RawDetection], threshold: float
) -> tuple[list[RawDetection], float]:
    """Filter raw detections by score; returns (kept, applied_threshold).

    The applied threshold travels with the result so runs are reproducible.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"confidence threshold must be in [0,1], got {threshold}")
    return [r for r in raw if r.score >= threshold], threshold
