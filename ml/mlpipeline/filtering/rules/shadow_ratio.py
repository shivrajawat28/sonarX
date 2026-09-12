"""shadow_ratio rule: dark-region adjacency cue. EXPERIMENTAL (Section 9.3, OPEN).

Disabled by default in config; effectiveness unproven until a real sonar dataset
has been reviewed. Kept because the shadow cue is physically motivated.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from mlpipeline.datatypes.detection import Detection
from mlpipeline.filtering.base import FilterContext, Verdict
from mlpipeline.filtering.rule_registry import register_rule


@register_rule
class ShadowRatioRule:
    name = "shadow_ratio"

    def evaluate(self, detection: Detection, ctx: FilterContext, params: dict[str, Any]) -> Verdict:
        dark_p = float(params.get("dark_percentile", 15))
        min_shadow = float(params.get("min_shadow_frac", 0.05))
        if ctx.pixels is None:
            return Verdict(rule=self.name, penalty=0.0)

        b = detection.bbox_source_coords.clipped(ctx.image_width, ctx.image_height)
        # search strip below/right of the box (sonar shadows fall off-axis)
        x0, y0 = int(b.x), int(b.y2)
        x1, y1 = int(b.x2), min(ctx.image_height, y0 + max(4, int(b.h)))
        if y1 <= y0 or x1 <= x0:
            return Verdict(rule=self.name, penalty=0.0)

        strip = ctx.pixels[y0:y1, x0:x1]
        threshold = float(np.percentile(ctx.pixels, dark_p))
        shadow_frac = float((strip < threshold).mean())
        if shadow_frac >= min_shadow:
            # a strong shadow next to a bright return is a GOOD sign: no penalty
            return Verdict(rule=self.name, penalty=0.0)
        return Verdict(
            rule=self.name,
            penalty=1.0,
            triggered=True,
            reason=f"shadow_ratio: shadow_frac {shadow_frac:.3f} < {min_shadow}",
        )
