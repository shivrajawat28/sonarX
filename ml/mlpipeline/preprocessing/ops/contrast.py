"""CLAHE contrast enhancement — the standard win on sonar texture (Section 7.2)."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from mlpipeline.preprocessing.base import OpContext
from mlpipeline.preprocessing.registry import register_op


@register_op
class ClaheOp:
    """Contrast-Limited Adaptive Histogram Equalization.

    Input float images in [0,1] are scaled to uint8 internally; output stays
    float32 in [0,1] so downstream ops see a consistent range.
    """

    name = "clahe"

    def apply(self, img: np.ndarray, params: dict[str, Any], ctx: OpContext) -> np.ndarray:
        clip = float(params.get("clip_limit", 2.0))
        tile = params.get("tile_grid_size", [8, 8])
        tile = (int(tile[0]), int(tile[1]))
        if clip <= 0:
            raise ValueError(f"clahe: clip_limit must be > 0, got {clip}")

        if img.ndim == 3 and img.shape[2] == 3:
            work = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ctx.warnings.append("clahe: 3-channel input collapsed to grayscale")
        else:
            work = img

        if work.dtype != np.uint8:
            work_u8 = np.clip(np.asarray(work, dtype=np.float64), 0.0, 1.0)
            work_u8 = (work_u8 * 255.0).round().astype(np.uint8)
        else:
            work_u8 = work

        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
        out = clahe.apply(work_u8)
        return out.astype(np.float32) / 255.0
