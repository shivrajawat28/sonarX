"""class_specific rule: per-class parameter table (Section 9.3).

Params shape: {"class_name": {"max_aspect_ratio": 12.0, "min_area_frac": 0.001}, ...}
Populated AFTER the class list is finalized (OPEN decision #1). Unknown classes
in the table are ignored (config drift must not crash serving).
"""
from __future__ import annotations

from typing import Any

from mlpipeline.datatypes.detection import Detection
from mlpipeline.filtering.base import FilterContext, Verdict
from mlpipeline.filtering.rule_registry import register_rule


@register_rule
class ClassSpecificRule:
    name = "class_specific"

    def evaluate(self, detection: Detection, ctx: FilterContext, params: dict[str, Any]) -> Verdict:
        table = params.get("per_class", {})
        cls_params = table.get(detection.class_name, {})
        if not cls_params:
            return Verdict(rule=self.name, penalty=0.0)

        b = detection.bbox_source_coords
        image_area = max(ctx.image_width * ctx.image_height, 1)
        area_frac = b.area / image_area

        max_ratio = cls_params.get("max_aspect_ratio")
        if max_ratio is not None and b.aspect_ratio > float(max_ratio):
            return Verdict(
                rule=self.name, penalty=1.0, triggered=True,
                reason=f"class_specific[{detection.class_name}]: aspect {b.aspect_ratio:.1f} > {max_ratio}",
            )
        min_area = cls_params.get("min_area_frac")
        if min_area is not None and area_frac < float(min_area):
            return Verdict(
                rule=self.name, penalty=1.0, triggered=True,
                reason=f"class_specific[{detection.class_name}]: area_frac {area_frac:.5f} < {min_area}",
            )
        max_area = cls_params.get("max_area_frac")
        if max_area is not None and area_frac > float(max_area):
            return Verdict(
                rule=self.name, penalty=1.0, triggered=True,
                reason=f"class_specific[{detection.class_name}]: area_frac {area_frac:.3f} > {max_area}",
            )
        return Verdict(rule=self.name, penalty=0.0)
