"""Standalone sonar image reader (PNG/TIFF/JPEG/BMP) — MVP contract (Section 10.1A)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class ImageReadError(Exception):
    """Undecodable/unsupported image — maps to INVALID_IMAGE / UNSUPPORTED_FILE_TYPE."""


def read_image_array(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read a grayscale image; returns (array HxW uint8, metadata dict)."""
    p = Path(path)
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageReadError(f"unsupported image extension '{p.suffix}' for {p.name}")
    data = np.fromfile(p, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ImageReadError(f"image could not be decoded: {p.name}")
    if img.ndim == 3 and img.shape[2] >= 3:
        img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    meta = {
        "format": p.suffix.lower().lstrip("."),
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
    }
    return img, meta
