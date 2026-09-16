"""Step 7 + 10 tests: training smoke chain, evaluation, inference engine, batch.

Gate (Section 25, step 7): smoke train -> eval -> register works.
Gate (Section 25, step 10): CLI inference produces complete InferenceResult with hashes.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from mlpipeline.config import load_config_file
from mlpipeline.config.schemas import DetectionConfig, PreprocessConfig
from mlpipeline.detection.stub import StubDetector
from mlpipeline.datatypes.image import ProcessedSonarImage, ScaleFactors
from mlpipeline.inference import SonarInferenceEngine, run_survey_batch
from mlpipeline.inference.engine import ModelNotLoadedError
from mlpipeline.registry import EvalRunStore, ModelRegistry


def _processed():
    return ProcessedSonarImage(
        image_id="img_e", width=64, height=64, source_width=64, source_height=64,
        scale_factors=ScaleFactors(scale_x=1.0, scale_y=1.0), config_hash="ph",
    )


class TestEngine:
    def _engine(self, run_filtering=True):
        cfg, h = load_config_file(Path("ml/configs/preprocessing/baseline_sonar.yaml"), PreprocessConfig)
        cfg = cfg.model_copy(deep=True)
        # force deterministic size for the tiny fixture
        for op in cfg.ops:
            if op.op == "resize_letterbox":
                op.params = {"target_size": [128, 128]}
        eng = SonarInferenceEngine(
            detector_kind="stub", registry=ModelRegistry(Path("/tmp/none.json")),
            preprocess_config=cfg, preprocess_hash=h, run_filtering=run_filtering,
        )
        eng.load("stub-e2e-v1")
        return eng

    def test_requires_explicit_preprocess_config(self):
        with pytest.raises(ValueError, match="preprocessing config"):
            SonarInferenceEngine(detector_kind="stub")

    def test_inference_before_load_raises(self):
        cfg, h = load_config_file(Path("ml/configs/preprocessing/baseline_sonar.yaml"), PreprocessConfig)
        eng = SonarInferenceEngine(detector_kind="stub", preprocess_config=cfg, preprocess_hash=h)
        with pytest.raises(ModelNotLoadedError):
            eng.run_array(np.zeros((64, 64), dtype=np.uint8))

    def test_end_to_end_result_is_complete(self):
        """Gate: complete InferenceResult with hashes + timings + provenance."""
        eng = self._engine()
        img = np.zeros((64, 64), dtype=np.uint8)
        img[20:40, 20:40] = 240  # bright target
        result = eng.run_array(img, image_id="img_e2e")

        assert result.model_version == "stub-e2e-v1"
        assert len(result.preprocess_config_hash) == 64
        assert result.filter_config_hash is not None
        assert len(result.detections) > 0
        assert result.timings_ms.preprocess_ms >= 0 and result.timings_ms.inference_ms >= 0
        d = result.detections[0]
        assert d.detection_id.startswith("det_")
        assert d.model_version == "stub-e2e-v1"
        assert d.preprocess_config_hash == result.preprocess_config_hash
        assert d.filtering_status in ("accepted", "flagged", "rejected")
        # stub detector is honest about being a fixture:
        assert d.class_name in ("stub_class_a", "stub_class_b")

    def test_standalone_image_geo_is_null_with_warning(self):
        eng = self._engine()
        result = eng.run_array(np.zeros((64, 64), dtype=np.uint8))
        for d in result.detections:
            assert d.latitude is None and d.longitude is None
            assert d.geo_status == "unavailable"
        assert any("coordinates unavailable" in w for w in result.warnings)

    def test_deterministic_given_same_input(self):
        eng = self._engine()
        img = np.zeros((64, 64), dtype=np.uint8)
        img[20:40, 20:40] = 240
        r1 = eng.run_array(img)
        r2 = eng.run_array(img)
        a = [(d.class_name, d.model_confidence, d.bbox_source_coords.as_list()) for d in r1.detections]
        b = [(d.class_name, d.model_confidence, d.bbox_source_coords.as_list()) for d in r2.detections]
        assert a == b

    def test_filter_on_vs_off_both_produce_detections(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[20:40, 20:40] = 240
        off = self._engine(run_filtering=False).run_array(img)
        on = self._engine(run_filtering=True).run_array(img)
        assert len(off.detections) == len(on.detections)  # annotate, never delete

    def test_batch_uses_same_engine(self, tmp_path):
        eng = self._engine()
        tmp = tmp_path / "sonar_batch_check"
        tmp.mkdir(exist_ok=True)
        paths = []
        for i, seed in enumerate([1, 2, 3]):
            rng = np.random.default_rng(seed)
            img = (rng.random((64, 64)) * 255).astype(np.uint8)
            p = tmp / f"tile_{i}.png"
            import cv2

            cv2.imwrite(str(p), img)
            paths.append(p)
        progress = []

        def cb(done, total, current):
            progress.append((done, total, current))

        results, errors = run_survey_batch(eng, paths, on_progress=cb)
        assert len(results) == 3 and not errors
        assert progress[-1] == (3, 3, "tile_2")
        # each result came through the identical pipeline (same model + hashes)
        assert len({r.model_version for r in results}) == 1
        assert len({r.preprocess_config_hash for r in results}) == 1


class TestTrainEvalRegisterChain:
    """Smoke chain: stub 'training' -> registry -> evaluation record (Step 7 gate)."""

    def _dataset(self, tmp_path: Path):
        import cv2

        root = tmp_path / "ds"
        (root / "images" / "train").mkdir(parents=True)
        (root / "images" / "val").mkdir(parents=True)
        (root / "images" / "test").mkdir(parents=True)
        (root / "labels" / "train").mkdir(parents=True)
        (root / "labels" / "val").mkdir(parents=True)
        (root / "labels" / "test").mkdir(parents=True)
        rng = np.random.default_rng(5)
        for split in ("train", "val", "test"):
            for i in range(4):
                name = f"{split}{i}.png"
                cv2.imwrite(str(root / "images" / split / name),
                            (rng.random((48, 48)) * 255).astype(np.uint8))
                (root / "labels" / split / f"{split}{i}.txt").write_text("0 0.5 0.5 0.3 0.3\n")
        return root

    def _manifest(self, tmp_path: Path, root: Path):
        from mlpipeline.datasets import build_manifest, save_manifest

        m = build_manifest(root, name="smoke", class_names=["alpha"], split_seed=42)
        return save_manifest(m, tmp_path / "manifests" / "smoke.json"), m

    def test_train_register_eval_smoke(self, tmp_path: Path):
        from mlpipeline.training.train import train_model

        root = self._dataset(tmp_path)
        manifest_path, manifest = self._manifest(tmp_path, root)

        # dataset config consumed by training (classes from data evidence)
        import yaml

        ds_cfg_path = tmp_path / "ds.yaml"
        ds_cfg_path.write_text(yaml.safe_dump({
            "name": "smoke", "root": str(root), "format": "yolo",
            "candidate_classes": ["alpha"], "split_seed": 42,
            "split_fractions": {"train": 0.7, "val": 0.2, "test": 0.1},
        }))

        reg = ModelRegistry(tmp_path / "registry.json")
        entry = train_model(
            "stub-smoke-v1", ds_cfg_path, "ml/configs/preprocessing/baseline_sonar.yaml",
            backend="stub", registry=reg, models_root=tmp_path / "models",
        )
        assert entry.class_map == {"0": "alpha"}  # from dataset config
        assert entry.status == "shadow"  # not active until evaluated
        assert entry.architecture_family == "stub"

        # evaluation with the stub (plumbing test — NOT a quality claim)
        store = EvalRunStore(tmp_path / "eval")
        from mlpipeline.evaluation.evaluate import evaluate_model

        run = evaluate_model(
            "stub-smoke-v1", manifest_path, split="test",
            registry=reg, store=store,
            notes="SMOKE TEST ONLY — plumbing check, no detection-quality claim",
        )
        assert run.model_version == "stub-smoke-v1"
        assert run.split == "test"
        assert run.preprocess_config_hash == entry.preprocess_config_ref.sha256
        assert "alpha" in run.per_class
        assert run.metrics.precision is not None

        # eval run is immutable
        with pytest.raises(FileExistsError):
            store.save(run)


class TestOperatingPointOverride:
    """Regression: a per-request confidence override must reach post-processing.

    The detector surfaces candidates below the default threshold, but the
    post-processor separately re-applies a confidence threshold. Passing the
    engine's DEFAULT config there discarded every recovered candidate, so the
    override was silently a no-op: identical output at 0.25 and at 0.05. The
    fix resolves one effective config and gives it to BOTH stages.

    The stub detector emits fixed scores 0.55/0.63/0.71/0.79, so this is a
    deterministic proof that the resolved threshold (not the default) governs
    post-processing.
    """

    def _engine_with_default(self, default_threshold: float):
        cfg, h = load_config_file(Path("ml/configs/preprocessing/baseline_sonar.yaml"), PreprocessConfig)
        cfg = cfg.model_copy(deep=True)
        for op in cfg.ops:
            if op.op == "resize_letterbox":
                op.params = {"target_size": [128, 128]}
        eng = SonarInferenceEngine(
            detector_kind="stub", registry=ModelRegistry(Path("/tmp/none.json")),
            preprocess_config=cfg, preprocess_hash=h, run_filtering=False,
            detection_config=DetectionConfig(confidence_threshold=default_threshold, iou_threshold=0.45),
        )
        eng.load("stub-e2e-v1")
        return eng

    def _image(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[10:30, 10:30] = 200
        img[40:60, 40:60] = 255
        return img

    def test_override_below_default_actually_surfaces_candidates(self):
        from mlpipeline.detection.base import PredictParams

        eng = self._engine_with_default(0.75)
        default = eng.run_array(self._image())
        lowered = eng.run_array(self._image(), params=PredictParams(confidence_threshold=0.5))

        # Without the fix these were identical: the post-processor re-applied 0.75.
        assert len(lowered.detections) > len(default.detections), (
            "override produced no extra detections — it is being discarded by "
            "post-processing (the engine must pass the RESOLVED config there)"
        )
        assert min(d.model_confidence for d in lowered.detections) < 0.75
        assert all(d.model_confidence >= 0.5 for d in lowered.detections)

    def test_applied_threshold_is_reported(self):
        from mlpipeline.detection.base import PredictParams

        eng = self._engine_with_default(0.75)
        r = eng.run_array(self._image(), params=PredictParams(confidence_threshold=0.5))
        # The run must state the threshold it actually used, not merely the default.
        assert r.applied_confidence_threshold == 0.5
        assert r.overrides_applied == {"confidence_threshold": 0.5}

    def test_no_override_uses_and_reports_the_default(self):
        eng = self._engine_with_default(0.75)
        r = eng.run_array(self._image())
        assert r.applied_confidence_threshold == 0.75
        assert r.overrides_applied == {}


class TestPredictImageCLI:
    def test_cli_produces_result_json(self, tmp_path: Path):
        import cv2
        import subprocess
        import sys

        img_path = tmp_path / "cli_img.png"
        cv2.imwrite(str(img_path), np.full((64, 64), 128, dtype=np.uint8))
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = "ml" + os.pathsep + "."
        r = subprocess.run(
            [sys.executable, "-m", "ml.scripts.predict_image",
             "--image", str(img_path), "--out", str(tmp_path / "result.json")],
            capture_output=True, text=True, env=env,
        )
        assert r.returncode == 0, r.stderr[-400:]
        payload = json.loads((tmp_path / "result.json").read_text())
        assert payload["model_version"].startswith("stub-")
        assert "preprocess_config_hash" in payload and "detections" in payload
