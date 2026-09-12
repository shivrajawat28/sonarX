"""min_size rule: tiny boxes are usually seafloor texture noise (Section 9.3)."""
from __future__ import annotations

from typing import Any

from mlpipeline.datatypes.detection import Detection
from mlpipeline.filtering.base import FilterContext, Verdict
from mlpipeline.filtering.rule_registry import register_rule


@register_rule
class MinSizeRule:
    name = "min_size"

    def evaluate(self, detection: Detection, ctx: FilterContext, params: dict[str, Any]) -> Verdict:
        min_area_frac = float(params.get("min_area_frac", 0.0004))
        image_area = max(ctx.image_width * ctx.image_height, 1)
        area_frac = detection.bbox_source_coords.area / image_area
        if area_frac < min_area_frac:
            return Verdict(
                rule=self.name,
                penalty=1.0,
                triggered=True,
                reason=f"min_size: area_frac {area_frac:.5f} < {min_area_frac}",
            )
        return Verdict(rule=self.name, penalty=0.0)
