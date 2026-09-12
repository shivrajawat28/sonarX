"""False-positive filtering (Section 9): deterministic rules + additive scoring.

Contract: receives canonical detections, returns the SAME detections annotated
with final_confidence / filtering_status / filter_reasons. Never deletes.
"""
from mlpipeline.filtering.base import FilterRule, FilterContext, Verdict
from mlpipeline.filtering.pipeline import FilterPipeline, FilteredResult
from mlpipeline.filtering.scorer import score_detection

__all__ = ["FilterRule", "FilterContext", "Verdict", "FilterPipeline", "FilteredResult", "score_detection"]
