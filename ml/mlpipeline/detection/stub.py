"""Stub detector — deterministic, for software tests and pipeline development ONLY.

HONESTY CONTRACT (README / architecture Section 25 / user constraint #4):
- This is NOT an AI detector. It produces deterministic pseudo-detections from
  image statistics so the pipeline plumbing is testable without weights.
- Its ModelMeta is marked is_stub=True and every result produced through it must
  be labeled as stub/test output. Never present stub output as real detection.
"""
from __future__ import annotations

import numpy as np

from mlpipeline.config.schemas import DetectionConfig
from mlpipeline.detection.base import (
    DetectorLoadError,
    DetectorPredictError,
    ModelMeta,
    PredictParams,
)
from mlpipeline.detection.registry import register_detector
from mlpipeline.datatypes.detection import BBox, RawDetection
from mlpipeline.datatypes.image import ProcessedSonarImage


@register_detector("stub")
class StubDetector:
    """Deterministic grid-scan 'detector'.

    Finds the brightest local maximum in each quadrant of the image and emits a
    fixed-size box around it with a fixed score. Same image -> same detections,
    which is exactly what unit/integration tests need.
    """

    name = "stub"

    def __init__(self) -> None:
        self._meta: ModelMeta | None = None

    def load(self, model_version: str) -> ModelMeta:
        if not model_version.startswith("stub"):
            raise DetectorLoadError(
                f"StubDetector can only load 'stub-*' model versions, got '{model_version}'"
            )
        # Class ids map to stable placeholder names; real classes come from real
        # model registry entries only.
        self._meta = ModelMeta(
            model_version=model_version,
            architecture_family="stub",
            class_map={0: "stub_class_a", 1: "stub_class_b"},
            input_size=[640, 640],
            framework="numpy",
        )
        return self._meta

    @property
    def is_loaded(self) -> bool:
        return self._meta is not None

    def metadata(self) -> ModelMeta:
        if self._meta is None:
            raise DetectorLoadError("StubDetector.load() not called")
        return self._meta

    def predict(
        self,
        image: ProcessedSonarImage,
        pixels: np.ndarray,
        params: PredictParams | None,
        defaults: DetectionConfig,
    ) -> list[RawDetection]:
        if self._meta is None:
            raise DetectorLoadError("StubDetector.load() not called")
        try:
            cfg = (params or PredictParams()).resolve(defaults)
            work = np.asarray(pixels)
            if work.ndim == 3:
                work = work[:, :, 0]
            h, w = work.shape
            quads = [(0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2),
                     (0, h // 2, w // 2, h), (w // 2, h // 2, w, h)]
            out: list[RawDetection] = []
            for qi, (x0, y0, x1, y1) in enumerate(quads):
                region = work[y0:y1, x0:x1]
                if region.size == 0:
                    continue
                peak = np.unravel_index(int(np.argmax(region)), region.shape)
                px, py = x0 + int(peak[1]), y0 + int(peak[0])
                box_w = max(8, w // 20)
                box_h = max(8, h // 20)
                # deterministic per-quadrant class + score
                score = round(min(0.95, 0.55 + 0.08 * qi), 4)
                cid = qi % 2
                box = BBox(
                    x=float(max(0, px - box_w // 2)),
                    y=float(max(0, py - box_h // 2)),
                    w=float(box_w),
                    h=float(box_h),
                ).clipped(image.width, image.height)
                out.append(RawDetection(class_id=cid, score=score, box=box))
            return out[: cfg.max_detections]
        except DetectorPredictError:
            raise
        except Exception as e:  # noqa: BLE001 — normalized to DetectorPredictError
            raise DetectorPredictError(f"stub predict failed: {e}") from e
