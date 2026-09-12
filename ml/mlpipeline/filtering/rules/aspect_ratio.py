"""aspect_ratio rule: extreme box ratios usually shadows/artifacts (Section 9.3)."""
from __future__ import annotations

from typing import Any

from mlpipeline.datatypes.detection import Detection
from mlpipeline.filtering.base import FilterContext, Verdict
from mlpipeline.filtering.rule_registry import register_rule


@register_rule
class AspectRatioRule:
    name = "aspect_ratio"

    def evaluate(self, detection: Detection, ctx: FilterContext, params: dict[str, Any]) -> Verdict:
        max_ratio = float(params.get("max_ratio", 8.0))
        ratio = detection.bbox_source_coords.aspect_ratio
        if ratio > max_ratio:
            return Verdict(
                rule=self.name,
                penalty=1.0,
                triggered=True,
                reason=f"aspect_ratio: {ratio:.1f} > {max_ratio}",
            )
        return Verdict(rule=self.name, penalty=0.0)
