"""Sonar-safe augmentation policy (train-only; Section 6.2).

Sonar is grayscale/intensity data: NO hue/saturation tricks. Allowed: flips,
small rotations, noise/gain jitter, mosaic-like tiling. These helpers document
the policy and provide reference implementations for the training wrapper.
"""
from __future__ import annotations

import numpy as np


def hflip(img: np.ndarray, boxes_xywh: list[list[float]] | None = None):
    """Horizontal flip; boxes [x,y,w,h] transformed accordingly."""
    out = np.fliplr(img).copy()
    if boxes_xywh is None:
        return out, None
    w_img = img.shape[1]
    new_boxes = []
    for x, y, bw, bh in boxes_xywh:
        new_boxes.append([w_img - x - bw, y, bw, bh])
    return out, new_boxes


def vflip(img: np.ndarray, boxes_xywh: list[list[float]] | None = None):
    out = np.flipud(img).copy()
    if boxes_xywh is None:
        return out, None
    h_img = img.shape[0]
    new_boxes = []
    for x, y, bw, bh in boxes_xywh:
        new_boxes.append([x, h_img - y - bh, bw, bh])
    return out, new_boxes


def gain_jitter(img: np.ndarray, gain: float = 0.1, bias: float = 0.05, seed: int | None = None):
    """Multiplicative gain + additive bias jitter in [0,1] intensity space."""
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32) * (1.0 + rng.uniform(-gain, gain)) + rng.uniform(-bias, bias)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def speckle_noise(img: np.ndarray, sigma: float = 0.03, seed: int | None = None):
    """Additive speckle-like noise (multiplicative approximation)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) * (1.0 + noise), 0.0, 1.0).astype(np.float32)


FORBIDDEN_AUGS = ("hue_shift", "saturation_jitter", "color_jitter", "channel_shuffle")
"""Banned for sonar data — asserted in the training wrapper."""
