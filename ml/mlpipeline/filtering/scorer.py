"""Scorer: additive penalty aggregation (Section 9.2).

final_confidence = model_confidence - sum(penalties), floored at 0.
Model confidence itself is never modified (ADR-006).
"""
from __future__ import annotations

from mlpipeline.datatypes.detection import Detection
from mlpipeline.filtering.base import Verdict


def score_detection(detection: Detection, verdicts: list[Verdict]) -> tuple[float, list[str]]:
    """Apply verdicts to a detection copy-source; returns (final_confidence, reasons)."""
    penalties = [v.penalty for v in verdicts if v.triggered]
    reasons = [v.reason for v in verdicts if v.triggered and v.reason]
    final = detection.model_confidence - sum(penalties)
    return max(0.0, min(final, 1.0)), reasons


def status_for(final_confidence: float, accept_threshold: float, flag_threshold: float) -> str:
    """Section 9.4 policy: >= accept -> accepted; >= flag -> flagged; else rejected."""
    if final_confidence >= accept_threshold:
        return "accepted"
    if final_confidence >= flag_threshold:
        return "flagged"
    return "rejected"
