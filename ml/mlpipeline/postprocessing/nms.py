"""NMS: explicit, testable non-maximum suppression (Section 12 postprocessing)."""
from __future__ import annotations

from mlpipeline.datatypes.detection import BBox


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two boxes."""
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def nms_numpy(boxes: list[BBox], scores: list[float], iou_threshold: float) -> list[int]:
    """Greedy NMS. Returns kept indices sorted by score (descending).

    Pure-python on purpose: tiny N at serving time, trivially testable.
    """
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    for idx in order:
        if all(iou(boxes[idx], boxes[k]) <= iou_threshold for k in keep):
            keep.append(idx)
    return keep
