"""Step 6 tests: detector abstraction.

Gate (Section 25, step 6): stub detector tests pass; registry append-only holds;
YOLO adapter degrades cleanly (DetectorLoadError) when weights/stack are absent.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from mlpipeline.config.schemas import DetectionConfig
from mlpipeline.detection import PredictParams, create_detector, known_detector_kinds
from mlpipeline.detection.base import DetectorLoadError, DetectorPredictError
from mlpipeline.detection.stub import StubDetector

# Adapters self-register on import; import explicitly so kind lists are complete
# regardless of import order elsewhere.
import mlpipeline.detection.yolo.adapter as _yolo_adapter  # noqa: F401
import mlpipeline.detection.onnx_adapter as _onnx_adapter  # noqa: F401
from mlpipeline.datatypes.image import ProcessedSonarImage, ScaleFactors
from mlpipeline.registry import ModelRegistry
from mlpipeline.datatypes.model import ModelVersion


@pytest.fixture
def loaded_stub():
    d = StubDetector()
    meta = d.load("stub-test-v0")
    return d, meta


@pytest.fixture
def processed_image():
    return ProcessedSonarImage(
        image_id="img_t", width=64, height=64, source_width=64, source_height=64,
        scale_factors=ScaleFactors(scale_x=1.0, scale_y=1.0), config_hash="h",
    )


class TestStubDetector:
    def test_load_rejects_non_stub_versions(self):
        with pytest.raises(DetectorLoadError, match="stub-"):
            StubDetector().load("yolo-real-v1")

    def test_metadata_available_after_load(self, loaded_stub):
        d, meta = loaded_stub
        assert d.is_loaded
        assert d.metadata() is meta
        assert meta.model_version == "stub-test-v0"
        assert 0 in meta.class_map  # names from metadata, not hardcoded callers

    def test_predict_without_load_raises(self, processed_image):
        with pytest.raises(DetectorLoadError):
            StubDetector().predict(
                processed_image, np.zeros((64, 64)), None, DetectionConfig()
            )

    def test_deterministic_predictions(self, loaded_stub, processed_image):
        d, _ = loaded_stub
        img = np.zeros((64, 64), dtype=np.float32)
        img[10:20, 10:20] = 1.0
        img[40:50, 40:50] = 0.8
        r1 = d.predict(processed_image, img, None, DetectionConfig())
        r2 = d.predict(processed_image, img, None, DetectionConfig())
        assert r1 == r2
        assert all(isinstance(r, object) for r in r1) and r1  # non-empty
        assert all(0 <= r.score <= 1 for r in r1)

    def test_predict_params_threshold_override_respected(self, loaded_stub, processed_image):
        d, _ = loaded_stub
        img = np.zeros((64, 64), dtype=np.float32)
        img[10:20, 10:20] = 1.0
        low = d.predict(processed_image, img, PredictParams(max_detections=1), DetectionConfig())
        assert len(low) == 1

    def test_boxes_within_processed_bounds(self, loaded_stub, processed_image):
        d, _ = loaded_stub
        rng = np.random.default_rng(0)
        out = d.predict(processed_image, rng.random((64, 64)).astype(np.float32), None, DetectionConfig())
        for r in out:
            assert 0 <= r.box.x and r.box.x2 <= processed_image.width + 1e-6
            assert 0 <= r.box.y and r.box.y2 <= processed_image.height + 1e-6


class TestDetectorRegistry:
    def test_kinds_include_stub_and_yolo(self):
        kinds = known_detector_kinds()
        assert "stub" in kinds and "yolo" in kinds and "onnx" in kinds

    def test_unknown_kind_clear_error(self):
        with pytest.raises(KeyError, match="stub"):
            create_detector("stub2_typo")

    def test_create_stub_returns_working_detector(self, processed_image):
        d = create_detector("stub")
        d.load("stub-reg-v1")
        out = d.predict(processed_image, np.zeros((32, 32), dtype=np.float32), None, DetectionConfig())
        assert isinstance(out, list)


class TestModelRegistry:
    def _entry(self, version="yolo-test-v1", family="yolo", status="shadow"):
        return ModelVersion(
            model_version=version, architecture_family=family,
            checkpoint_path=f"models/weights/{version}/best.pt",
            class_map={"0": "alpha", "1": "beta"},
            preprocess_config_ref={"path": "ml/configs/preprocessing/baseline_sonar.yaml", "sha256": "x"},
            status=status,
        )

    def test_register_and_get(self, tmp_path: Path):
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(self._entry())
        got = reg.get("yolo-test-v1")
        assert got is not None and got.class_map["1"] == "beta"

    def test_append_only_no_silent_overwrite(self, tmp_path: Path):
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(self._entry())
        with pytest.raises(Exception, match="already registered"):
            reg.register(self._entry())  # same id -> refused

    def test_status_transition_preserves_entry(self, tmp_path: Path):
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(self._entry())
        reg.set_status("yolo-test-v1", "active")
        assert reg.get_active().model_version == "yolo-test-v1"
        assert reg.get("yolo-test-v1").class_map == {"0": "alpha", "1": "beta"}

    def test_file_format_is_json_models_list(self, tmp_path: Path):
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(self._entry())
        raw = json.loads((tmp_path / "registry.json").read_text())
        assert isinstance(raw["models"], list) and raw["models"][0]["model_version"] == "yolo-test-v1"


class TestYoloAdapterDegradesCleanly:
    # NOTE: in this environment torch/ultralytics are NOT installed, so load()
    # fails at the stack check first. These tests assert the clean
    # DetectorLoadError contract regardless of which guard fires.

    def _entry(self, version="yolo-test-v1", family="yolo", status="shadow"):
        return ModelVersion(
            model_version=version, architecture_family=family,
            checkpoint_path=f"models/weights/{version}/best.pt",
            class_map={"0": "alpha", "1": "beta"},
            preprocess_config_ref={"path": "ml/configs/preprocessing/baseline_sonar.yaml", "sha256": "x"},
            status=status,
        )

    def test_load_missing_version_raises(self):
        from mlpipeline.detection.yolo.adapter import YOLODetector

        reg = ModelRegistry(Path("/tmp/does-not-exist-registry.json"))
        with pytest.raises(DetectorLoadError):
            YOLODetector(registry=reg).load("never-registered-v9")

    def test_load_missing_weights_raises(self, tmp_path: Path):
        pytest.importorskip("torch", reason="stack-dependent guard order")
        from mlpipeline.detection.yolo.adapter import YOLODetector

        reg = ModelRegistry(tmp_path / "r.json")
        reg.register(self._entry(version="yolo-ghost-v1"))
        with pytest.raises(DetectorLoadError, match="checkpoint missing"):
            YOLODetector(registry=reg).load("yolo-ghost-v1")

    def test_load_non_yolo_family_raises(self, tmp_path: Path):
        pytest.importorskip("torch", reason="stack-dependent guard order")
        from mlpipeline.detection.yolo.adapter import YOLODetector

        reg = ModelRegistry(tmp_path / "r.json")
        reg.register(self._entry(version="future-det-v1", family="transformer"))
        with pytest.raises(DetectorLoadError, match="transformer"):
            YOLODetector(registry=reg).load("future-det-v1")
