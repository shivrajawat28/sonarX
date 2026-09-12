"""Letterbox resize: model-input sizing with recorded scale/pad (Section 6.6).

The scale factors written into ``ctx.extras`` are the ONLY mechanism by which
post-processing maps boxes back to source coordinates — exactly once.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from mlpipeline.preprocessing.base import OpContext
from mlpipeline.preprocessing.registry import register_op


@register_op
class ResizeLetterboxOp:
    """Resize preserving aspect ratio, pad to target with a constant value.

    Records into ctx.extras:
        scale_x, scale_y, pad_x, pad_y, target_size
    """

    name = "resize_letterbox"

    def apply(self, img: np.ndarray, params: dict[str, Any], ctx: OpContext) -> np.ndarray:
        target = params.get("target_size", [640, 640])
        tw, th = int(target[0]), int(target[1])
        pad_value = float(params.get("pad_value", 0))
        if tw <= 0 or th <= 0:
            raise ValueError(f"resize_letterbox: invalid target_size {target}")

        h, w = img.shape[:2]
        scale = min(tw / w, th / h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        if img.ndim == 3:
            resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
        else:
            resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

        pad_x = (tw - new_w) / 2.0
        pad_y = (th - new_h) / 2.0

        if img.ndim == 3:
            canvas = np.full((th, tw, img.shape[2]), pad_value, dtype=resized.dtype)
        else:
            canvas = np.full((th, tw), pad_value, dtype=resized.dtype)
        x0 = int(round(pad_x))
        y0 = int(round(pad_y))
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized

        ctx.extras.update(
            {
                "scale_x": scale,
                "scale_y": scale,
                "pad_x": x0,
                "pad_y": y0,
                "target_size": [tw, th],
            }
        )
        return canvas
