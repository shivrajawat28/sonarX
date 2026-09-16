"""YOLODetector: concrete Detector via the ultralytics stack (Section 8.2).

- The framework import is lazy: only when a YOLO model is actually loaded.
- Class map comes from the registry entry (verbatim from model metadata) —
  never from the weights file alone, never hardcoded.
- Returns RawDetection in PROCESSED-image pixel space (protocol contract).
"""
from __future__ import annotations

from pathlib import Path

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
from mlpipeline.registry.models import ModelRegistry, get_registry, resolve_repo_path


@register_detector("yolo")
class YOLODetector:
    """ultralytics-YOLO adapter. kind='yolo' in the detector registry."""

    name = "yolo"

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self._registry = registry or get_registry()
        self._meta: ModelMeta | None = None
        self._model = None
        self._entry = None
        # Test-time augmentation. Off by default: it roughly doubles inference
        # cost, so it is opt-in and measured (see docs/EVALUATION.md).
        self._tta = False

    def set_tta(self, enabled: bool) -> None:
        """Enable/disable multi-scale+flip test-time augmentation."""
        self._tta = bool(enabled)

    @property
    def tta(self) -> bool:
        return self._tta

    def load(self, model_version: str) -> ModelMeta:
        try:
            import torch  # noqa: F401 — availability check only
            from ultralytics import YOLO  # lazy: heavy import
        except ImportError as e:
            raise DetectorLoadError(
                f"YOLO stack unavailable (install ml extras: torch/ultralytics): {e}"
            ) from e

        entry = self._registry.get(model_version)
        if entry is None:
            raise DetectorLoadError(f"model version '{model_version}' not in registry")
        if entry.architecture_family != "yolo":
            raise DetectorLoadError(
                f"model '{model_version}' is '{entry.architecture_family}', not yolo"
            )
        weights = resolve_repo_path(entry.checkpoint_path)
        if not weights.is_file():
            raise DetectorLoadError(f"checkpoint missing: {weights}")

        try:
            self._model = YOLO(str(weights))
        except Exception as e:  # noqa: BLE001 — any load failure = MODEL_UNAVAILABLE
            raise DetectorLoadError(f"failed to load YOLO weights {weights}: {e}") from e

        class_map = {int(k): v for k, v in entry.class_map.items()}
        self._meta = ModelMeta(
            model_version=entry.model_version,
            architecture_family=entry.architecture_family,
            class_map=class_map,
            input_size=list(entry.input_size),
            framework=entry.framework,
        )
        self._entry = entry
        return self._meta

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._meta is not None

    def metadata(self) -> ModelMeta:
        if self._meta is None:
            raise DetectorLoadError("YOLODetector.load() not called")
        return self._meta

    def predict(
        self,
        image: ProcessedSonarImage,
        pixels,
        params: PredictParams | None,
        defaults: DetectionConfig,
    ) -> list[RawDetection]:
        if not self.is_loaded:
            raise DetectorLoadError("YOLODetector.load() not called")
        cfg = (params or PredictParams()).resolve(defaults)
        try:
            results = self._model.predict(
                source=_to_model_input(pixels, self._meta.input_size),
                conf=cfg.confidence_threshold,
                iou=cfg.iou_threshold,
                max_det=cfg.max_detections,
                augment=self._tta,  # TTA: averaged multi-scale/flip inference
                verbose=False,
                device=None,  # framework default; backend sets DEVICE env globally
            )
        except Exception as e:  # noqa: BLE001 — normalized per protocol
            raise DetectorPredictError(f"YOLO inference failed: {e}") from e

        out: list[RawDetection] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.tolist() if hasattr(boxes, "xyxy") else []
            scores = boxes.conf.tolist() if hasattr(boxes, "conf") else []
            class_ids = boxes.cls.tolist() if hasattr(boxes, "cls") else []
            for (x1, y1, x2, y2), score, cid in zip(xyxy, scores, class_ids):
                out.append(
                    RawDetection(
                        class_id=int(cid),
                        score=float(score),
                        box=BBox(x=float(x1), y=float(y1), w=float(x2 - x1), h=float(y2 - y1)),
                    )
                )
        return out


def _to_model_input(pixels, input_size: list[int]):
    """Expand single-channel to 3-channel BGR-equivalent for YOLO compatibility.

    (Section 6.6: identity channel expansion; preprocessing already letterboxed.)
    """
    import numpy as np

    arr = np.asarray(pixels)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.max() <= 1.0 + 1e-6:
        arr = (arr * 255.0).round().astype(np.uint8)
    return arr
