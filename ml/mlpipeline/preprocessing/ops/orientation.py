"""Waterfall orientation fix (Section 7.2, OPEN decision #2).

Port/starboard orientation depends on the reader/sonar system; this op exists so
orientation can be corrected per dataset via config once formats are reviewed.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from mlpipeline.preprocessing.base import OpContext
from mlpipeline.preprocessing.registry import register_op


@register_op
class OptionalFlipWaterfallOp:
    """Flip the waterfall image horizontally (and/or vertically) per config."""

    name = "optional_flip_waterfall"

    def apply(self, img: np.ndarray, params: dict[str, Any], ctx: OpContext) -> np.ndarray:
        fh = bool(params.get("flip_horizontal", False))
        fv = bool(params.get("flip_vertical", False))
        if not fh and not fv:
            ctx.warnings.append(
                "optional_flip_waterfall enabled but no flip specified; image unchanged"
            )
            return img
        out = img
        if fv:
            out = np.flipud(out)
        if fh:
            out = np.fliplr(out)
        return np.ascontiguousarray(out)
