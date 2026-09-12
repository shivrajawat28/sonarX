"""PostProcessor: RawDetection[] -> canonical Detection[] (Section 6.3 step 4).

Responsibilities: threshold -> NMS -> rescale boxes to SOURCE coordinates ->
resolve class names from the model's class_map. Model confidence untouched.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from mlpipeline.config.schemas import DetectionConfig
from mlpipeline.detection.base import ModelMeta
from mlpipeline.datatypes.detection import BBox, Detection, RawDetection
from mlpipeline.datatypes.image import ProcessedSonarImage
from mlpipeline.postprocessing.nms import nms_numpy
from mlpipeline.postprocessing.thresholds import apply_confidence_threshold


class UnknownClassIdError(Exception):
    """Model emitted a class id absent from the registry class_map — refuses to guess."""


class PostprocessResult(BaseModel):
    detections: list[Detection] = Field(default_factory=list)
    applied_confidence_threshold: float = 0.0
    applied_iou_threshold: float = 0.0
    n_dropped_by_threshold: int = 0
    n_dropped_by_nms: int = 0


class PostProcessor:
    """Stateless; safe to reuse across images."""

    def process(
        self,
        raw: list[RawDetection],
        processed: ProcessedSonarImage,
        meta: ModelMeta,
        config: DetectionConfig,
        run_id: str | None = None,
        filter_config_hash: str | None = None,
    ) -> PostprocessResult:
        # 1) confidence threshold (recorded)
        kept, applied_thr = apply_confidence_threshold(raw, config.confidence_threshold)
        n_thr_dropped = len(raw) - len(kept)

        # 2) NMS per class-id (class-aware)
        keep_idx: list[int] = []
        for cid in {r.class_id for r in kept}:
            group = [(i, r) for i, r in enumerate(kept) if r.class_id == cid]
            idxs = nms_numpy([r.box for _, r in group], [r.score for _, r in group], config.iou_threshold)
            keep_idx.extend(group[i][0] for i in idxs)
        keep_idx.sort()
        nms_dropped = len(kept) - len(keep_idx)
        kept = [kept[i] for i in keep_idx]

        detections: list[Detection] = []
        for r in kept:
            cname = meta.class_map.get(r.class_id)
            if cname is None:
                raise UnknownClassIdError(
                    f"model '{meta.model_version}' produced class_id {r.class_id} "
                    f"which is absent from its registered class_map {meta.class_map}"
                )
            src = r.box  # processed-space box from the detector
            sx, sy = processed.to_source_coords(src.x, src.y)
            sx2, sy2 = processed.to_source_coords(src.x2, src.y2)
            src_box = BBox(x=sx, y=sy, w=max(sx2 - sx, 0.0), h=max(sy2 - sy, 0.0))
            detections.append(
                Detection(
                    run_id=run_id,
                    image_id=processed.image_id,
                    class_name=cname,
                    model_confidence=r.score,
                    final_confidence=r.score,  # filter stage adjusts this
                    bbox_source_coords=src_box,
                    bbox_processed_coords=r.box,
                    model_version=meta.model_version,
                    preprocess_config_hash=processed.config_hash,
                    filter_config_hash=filter_config_hash,
                )
            )

        return PostprocessResult(
            detections=detections,
            applied_confidence_threshold=applied_thr,
            applied_iou_threshold=config.iou_threshold,
            n_dropped_by_threshold=n_thr_dropped,
            n_dropped_by_nms=nms_dropped,
        )
