"""SonarInferenceEngine — the ONLY orchestration object for detection (Section 8.3).

The backend calls this; it never orchestrates ML stages itself. Batch mode
loops THIS engine (no second inference path, per user constraint #7).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mlpipeline.config.loader import load_config_file
from mlpipeline.config.schemas import DetectionConfig, FilterConfig, PreprocessConfig
from mlpipeline.detection.base import DetectorLoadError, ModelMeta, PredictParams
from mlpipeline.detection.registry import create_detector
from mlpipeline.datatypes.detection import Detection
from mlpipeline.datatypes.image import ProcessedSonarImage, ScaleFactors
from mlpipeline.datatypes.results import InferenceResult
from mlpipeline.filtering.pipeline import FilterPipeline
from mlpipeline.geolocation.associator import Geolocator
from mlpipeline.io.image_reader import read_image_array
from mlpipeline.postprocessing.decoder import PostProcessor
from mlpipeline.registry.models import ModelRegistry, get_registry


class ModelNotLoadedError(Exception):
    """Inference requested before a model loaded — maps to MODEL_UNAVAILABLE."""


class SonarInferenceEngine:
    """Configured once (detector kind + configs), then `run_array` per image."""

    def __init__(
        self,
        detector_kind: str,
        registry: ModelRegistry | None = None,
        preprocess_config: PreprocessConfig | None = None,
        preprocess_hash: str | None = None,
        detection_config: DetectionConfig | None = None,
        filter_config: FilterConfig | None = None,
        filter_config_hash: str | None = None,
        run_filtering: bool = True,
        preprocessing_engine=None,
    ) -> None:
        self.registry = registry or get_registry()
        self.detector_kind = detector_kind
        self.detector = create_detector(detector_kind)
        self._meta: ModelMeta | None = None

        if preprocess_config is None or preprocess_hash is None:
            raise ValueError(
                "SonarInferenceEngine requires an explicit preprocessing config + hash "
                "(loaded by hash from the model version's record — ADR-003)"
            )
        self.preprocess_config = preprocess_config
        self.preprocess_hash = preprocess_hash
        self.detection_config = detection_config or DetectionConfig()

        self.filter_pipeline: FilterPipeline | None = None
        if run_filtering:
            if filter_config is None:
                # default bundled config (still hash-recorded, never hidden behavior)
                from pathlib import Path as _P

                default_path = _P(__file__).resolve().parents[2] / "configs" / "filtering" / "rules.yaml"
                filter_config, filter_config_hash = load_config_file(default_path, FilterConfig)
            self.filter_pipeline = FilterPipeline(filter_config, filter_config_hash)
            self.filter_config_hash = filter_config_hash
        else:
            self.filter_config_hash = None

        self.post = PostProcessor()
        self._preprocessing_engine = preprocessing_engine
        self.geolocator = Geolocator(None)  # wired per-run with real tracks only

    # -- lifecycle ----------------------------------------------------------
    def load(self, model_version: str) -> ModelMeta:
        """Load the detector for a registry version; validates preprocessing parity."""
        self._meta = self.detector.load(model_version)
        return self._meta

    @property
    def loaded_version(self) -> str | None:
        return self._meta.model_version if self._meta else None

    # -- inference -----------------------------------------------------------
    def run_array(
        self,
        pixels: np.ndarray,
        image_id: str = "unknown",
        params: PredictParams | None = None,
        track=None,
        fraction_of_image: float | None = None,
    ) -> InferenceResult:
        """Run the full pipeline on an in-memory image array.

        pixels: grayscale uint8/float source-space image.
        """
        from mlpipeline.inference.tracing import StageTracer

        if self._meta is None:
            raise ModelNotLoadedError(
                "engine.load(model_version) must be called before inference"
            )

        tracer = StageTracer()
        with tracer.stage("read_ms"):
            source = _as_gray_float(pixels)

        with tracer.stage("preprocess_ms"):
            engine = self._preprocessing_engine
            if engine is None:
                from mlpipeline.preprocessing.pipeline import PreprocessingEngine

                engine = PreprocessingEngine()
            processed_meta, processed_arr = engine.run_with_array(
                source, self.preprocess_config, self.preprocess_hash, image_id
            )

        # ONE effective config for the whole run. A per-request override must
        # govern BOTH the detector and post-processing: the detector surfaces the
        # extra candidates, and the post-processor thresholds the SAME way. Passing
        # `self.detection_config` to the post-processor would re-apply the default
        # threshold and silently discard every candidate the override recovered,
        # making the override a no-op.
        effective = (params or PredictParams()).resolve(self.detection_config)

        with tracer.stage("inference_ms"):
            raw = self.detector.predict(processed_meta, processed_arr, params, self.detection_config)

        with tracer.stage("postprocess_ms"):
            post = self.post.process(
                raw, processed_meta, self._meta, effective,
                filter_config_hash=self.filter_config_hash,
            )
            detections: list[Detection] = post.detections

        if self.filter_pipeline is not None:
            with tracer.stage("filter_ms"):
                from mlpipeline.filtering.base import FilterContext

                lo, hi = np.percentile(source, [10, 90])
                ctx = FilterContext(
                    image_width=int(source.shape[1]),
                    image_height=int(source.shape[0]),
                    pixels=source,
                    image_intensity_p10=float(lo),
                    image_intensity_p90=float(hi),
                    class_map={int(k): v for k, v in self._meta.class_map.items()},
                )
                filtered = self.filter_pipeline.apply(detections, ctx)
                detections = filtered.detections

        with tracer.stage("geolocate_ms"):
            # Fresh geolocator per run: tracks come from the caller's real nav data.
            self.geolocator = Geolocator(track)
            detections = self.geolocator.associate(detections, fraction_of_image)

        warnings: list[str] = []
        if detections and all(not d.has_coordinates() for d in detections):
            warnings.append(
                "navigation metadata missing or unusable — coordinates unavailable (never fabricated)"
            )
        warnings.extend(processed_meta.warnings)

        return InferenceResult(
            image_id=image_id,
            model_version=self._meta.model_version,
            preprocess_config_hash=self.preprocess_hash,
            preprocess_config_name=self.preprocess_config.name,
            filter_config_hash=self.filter_config_hash,
            overrides_applied=(
                {"confidence_threshold": effective.confidence_threshold}
                if params is not None and params.confidence_threshold is not None
                else {}
            ),
            applied_confidence_threshold=effective.confidence_threshold,
            detections=detections,
            timings_ms=tracer.timings,
            warnings=warnings,
        )

    def run_image_file(
        self,
        path: str | Path,
        image_id: str | None = None,
        params: PredictParams | None = None,
        track=None,
        fraction_of_image: float | None = None,
    ) -> InferenceResult:
        """Convenience for CLI usage: read a file from disk then run_array."""
        arr, _meta = read_image_array(path)
        return self.run_array(
            arr, image_id=image_id or Path(path).stem, params=params, track=track,
            fraction_of_image=fraction_of_image,
        )


def _as_gray_float(pixels: np.ndarray) -> np.ndarray:
    arr = np.asarray(pixels)
    if arr.ndim == 3:
        arr = arr[:, :, 0]  # take first channel (for grayscale/sonar, channels are identical; strips alpha if RGBA)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
    return arr

