"""Optional ONNXRuntime detector adapter (Section 8.2, serve extra).

Kept behind the same `Detector` protocol; only usable when `onnxruntime` is
installed. Registration happens on import; nothing else in the app changes.
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
from mlpipeline.postprocessing.nms import nms_numpy
from mlpipeline.registry.models import ModelRegistry, get_registry, resolve_repo_path


@register_detector("onnx")
class OnnxDetector:
    """ONNXRuntime adapter: exported YOLO-style model (1xN, xyxy+score+class rows)."""

    name = "onnx"

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self._registry = registry or get_registry()
        self._meta: ModelMeta | None = None
        self._session = None

    def load(self, model_version: str) -> ModelMeta:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise DetectorLoadError(
                f"onnxruntime unavailable (pip install onnxruntime): {e}"
            ) from e

        entry = self._registry.get(model_version)
        if entry is None:
            raise DetectorLoadError(f"model version '{model_version}' not in registry")
        weights = resolve_repo_path(entry.checkpoint_path)
        if not weights.is_file():
            raise DetectorLoadError(f"checkpoint missing: {weights}")
        try:
            self._session = ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])
        except Exception as e:  # noqa: BLE001
            raise DetectorLoadError(f"failed to load ONNX model {weights}: {e}") from e
        self._meta = ModelMeta(
            model_version=entry.model_version,
            architecture_family=entry.architecture_family,
            class_map={int(k): v for k, v in entry.class_map.items()},
            input_size=list(entry.input_size),
            framework="onnxruntime",
        )
        return self._meta

    @property
    def is_loaded(self) -> bool:
        return self._session is not None and self._meta is not None

    def metadata(self) -> ModelMeta:
        if self._meta is None:
            raise DetectorLoadError("OnnxDetector.load() not called")
        return self._meta

    def predict(
        self,
        image: ProcessedSonarImage,
        pixels,
        params: PredictParams | None,
        defaults: DetectionConfig,
    ) -> list[RawDetection]:
        if not self.is_loaded:
            raise DetectorLoadError("OnnxDetector.load() not called")
        import numpy as np

        cfg = (params or PredictParams()).resolve(defaults)
        try:
            arr = np.asarray(pixels, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[None, None, :, :]  # 1x1xHxW
            elif arr.ndim == 3:
                arr = arr.transpose(2, 0, 1)[None]  # 1xCxHxW
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: arr})
        except Exception as e:  # noqa: BLE001
            raise DetectorPredictError(f"ONNX inference failed: {e}") from e

        rows = np.asarray(outputs[0]).reshape(-1, 6)  # x1,y1,x2,y2,score,class_id
        keep = rows[:, 4] >= cfg.confidence_threshold
        rows = rows[keep]
        boxes = [BBox(x=r[0], y=r[1], w=r[2] - r[0], h=r[3] - r[1]) for r in rows]
        scores = rows[:, 4].tolist()
        cids = rows[:, 5].astype(int).tolist()
        keep_idx = nms_numpy(boxes, scores, cfg.iou_threshold)
        return [
            RawDetection(class_id=cids[i], score=float(scores[i]), box=boxes[i])
            for i in keep_idx[: cfg.max_detections]
        ]
