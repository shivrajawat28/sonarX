"""intensity_outlier rule: region intensity contrast vs the image band (Section 9.3)."""
from __future__ import annotations

from typing import Any

import numpy as np

from mlpipeline.datatypes.detection import Detection
from mlpipeline.filtering.base import FilterContext, Verdict
from mlpipeline.filtering.rule_registry import register_rule


@register_rule
class IntensityOutlierRule:
    """Man-made objects often have distinctive return strength vs seafloor.

    If the detection region has near-zero INTERNAL contrast (flat texture,
    no highlight/shadow structure), it is likely texture noise. Measuring the
    region's own intensity spread avoids a prior bug where a region containing
    BOTH a bright return and its acoustic shadow averaged back into the image's
    central band and was wrongly penalized. Requires ctx.pixels; skips
    (penalty 0) when pixels are unavailable (e.g. metadata-only filtering).
    """

    name = "intensity_outlier"

    def evaluate(self, detection: Detection, ctx: FilterContext, params: dict[str, Any]) -> Verdict:
        # NOTE: percentile_band remains an accepted config key (compat) but the
        # structural signal for this rule is the region's INTERNAL spread.
        min_contrast = float(params.get("min_contrast", 0.02))
        if ctx.pixels is None:
            return Verdict(rule=self.name, penalty=0.0)  # cannot evaluate: skip silently

        b = detection.bbox_source_coords.clipped(ctx.image_width, ctx.image_height)
        x0, y0 = int(b.x), int(b.y)
        x1, y1 = max(x0 + 1, int(b.x2)), max(y0 + 1, int(b.y2))
        region = ctx.pixels[y0:y1, x0:x1]
        if region.size == 0:
            return Verdict(rule=self.name, penalty=0.0)

        # INTERNAL contrast: spread of pixel intensities inside the region.
        region_f = region.astype(np.float32)
        if region_f.max() > 1.0 + 1e-6:
            region_f = region_f / 255.0
        contrast = float(np.std(region_f))

        if contrast < min_contrast:
            return Verdict(
                rule=self.name,
                penalty=1.0,
                triggered=True,
                reason=f"intensity_outlier: contrast {contrast:.3f} < {min_contrast}",
            )
        return Verdict(rule=self.name, penalty=0.0)
