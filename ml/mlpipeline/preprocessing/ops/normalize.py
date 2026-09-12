"""Robust percentile-based intensity normalization (Section 7.2)."""
from __future__ import annotations

from typing import Any

import numpy as np

from mlpipeline.preprocessing.base import OpContext
from mlpipeline.preprocessing.registry import register_op


@register_op
class NormalizeIntensityOp:
    """Stretch intensities to [0, 1] using robust percentile bounds.

    Percentile clipping avoids single-pixel outliers dominating sonar contrast.
    """

    name = "normalize_intensity"

    def apply(self, img: np.ndarray, params: dict[str, Any], ctx: OpContext) -> np.ndarray:
        low_p = float(params.get("low_percentile", 1.0))
        high_p = float(params.get("high_percentile", 99.0))
        if not (0.0 <= low_p < high_p <= 100.0):
            raise ValueError(
                f"normalize_intensity: need 0 <= low_percentile < high_percentile <= 100, "
                f"got {low_p}/{high_p}"
            )

        work = img.astype(np.float64, copy=True)
        lo, hi = np.percentile(work, [low_p, high_p])
        if hi - lo < 1e-12:
            ctx.warnings.append(
                f"normalize_intensity: near-constant image (range {hi - lo:.2e}); output flat"
            )
            return np.zeros_like(work, dtype=np.float32)

        out = (work - lo) / (hi - lo)
        np.clip(out, 0.0, 1.0, out=out)
        return out.astype(np.float32)
