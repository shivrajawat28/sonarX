"""Fixtures: tiny synthetic sonar-like images. For unit tests ONLY — never real data.

The synthetic pattern (bright blob + shadow on textured background) exercises
normalization/CLAHE/letterbox without pretending to represent any real seafloor.
"""
from __future__ import annotations

import numpy as np
import pytest


def make_sonar_like_image(width: int = 200, height: int = 100, seed: int = 7) -> np.ndarray:
    """Deterministic grayscale 'sonar-like' test image in uint8 [0, 255].

    Pattern: mid-gray textured background, one bright blob (high-return object),
    one dark ellipse (acoustic shadow) — enough to exercise contrast/normalize ops.
    """
    rng = np.random.default_rng(seed)
    img = rng.integers(90, 110, size=(height, width), dtype=np.uint8).astype(np.float64)

    # bright blob
    cx, cy, r = int(width * 0.3), int(height * 0.5), 12
    yy, xx = np.mgrid[0:height, 0:width]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    img[mask] = 235.0

    # dark shadow ellipse to the right of the blob (acoustic-shadow-like)
    sx, sy = int(width * 0.3) + 25, int(height * 0.5)
    smask = ((xx - sx) / 18.0) ** 2 + ((yy - sy) / 6.0) ** 2 <= 1.0
    img[smask] = 15.0

    return img.round().astype(np.uint8)


@pytest.fixture
def sonar_like_image() -> np.ndarray:
    return make_sonar_like_image()
