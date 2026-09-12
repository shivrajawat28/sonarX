"""Denoising ops — disabled by default; measure before adopting (Section 7.2)."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from mlpipeline.preprocessing.base import OpContext
from mlpipeline.preprocessing.registry import register_op


@register_op
class DenoiseMedianOp:
    """Median blur: cheap speckle suppression. Kernel must be odd and >= 3."""

    name = "denoise_median"

    def apply(self, img: np.ndarray, params: dict[str, Any], ctx: OpContext) -> np.ndarray:
        k = int(params.get("kernel", 3))
        if k < 3 or k % 2 == 0:
            raise ValueError(f"denoise_median: kernel must be odd and >= 3, got {k}")
        if img.dtype != np.uint8:
            work = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)
            u8 = (work * 255.0).round().astype(np.uint8)
        else:
            u8 = img
        out = cv2.medianBlur(u8, k)
        return out.astype(np.float32) / 255.0 if img.dtype != np.uint8 else out


@register_op
class DenoiseNlmOp:
    """Non-local means denoising: stronger, much slower — experimental only."""

    name = "denoise_nlm"

    def apply(self, img: np.ndarray, params: dict[str, Any], ctx: OpContext) -> np.ndarray:
        h = float(params.get("h", 10.0))
        tw = int(params.get("template_window_size", 7))
        sw = int(params.get("search_window_size", 21))
        if img.dtype != np.uint8:
            work = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)
            u8 = (work * 255.0).round().astype(np.uint8)
        else:
            u8 = img
        out = cv2.fastNlMeansDenoising(u8, None, h=h, templateWindowSize=tw, searchWindowSize=sw)
        return out.astype(np.float32) / 255.0 if img.dtype != np.uint8 else out
