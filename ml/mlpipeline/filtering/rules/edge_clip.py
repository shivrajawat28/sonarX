"""edge_clip rule: border-clipped boxes can't be assessed — flag, don't reject hard."""
from __future__ import annotations

from typing import Any

from mlpipeline.datatypes.detection import Detection
from mlpipeline.filtering.base import FilterContext, Verdict
from mlpipeline.filtering.rule_registry import register_rule


@register_rule
class EdgeClipRule:
    name = "edge_clip"

    def evaluate(self, detection: Detection, ctx: FilterContext, params: dict[str, Any]) -> Verdict:
        tol = float(params.get("tolerance_px", 2.0))
        b = detection.bbox_source_coords
        touches = (
            b.x <= tol
            or b.y <= tol
            or b.x2 >= ctx.image_width - tol
            or b.y2 >= ctx.image_height - tol
        )
        if touches:
            return Verdict(
                rule=self.name,
                penalty=1.0,
                triggered=True,
                reason=f"edge_clip: box touches image border (tol {tol}px)",
            )
        return Verdict(rule=self.name, penalty=0.0)
