"""Step 4 acceptance tests: preprocessing engine.

Gate (Section 25, step 4): determinism test; ops behave; config-hash provenance.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for fixtures import
from fixtures import make_sonar_like_image  # noqa: E402

from mlpipeline.datatypes.image import ProcessedSonarImage  # noqa: E402

from mlpipeline.config import PreprocessConfig, load_config_file  # noqa: E402
from mlpipeline.datatypes.image import ScaleFactors  # noqa: E402
from mlpipeline.preprocessing import (  # noqa: E402
    PreprocessingEngine,
    PreprocessingError,
    known_op_names,
    register_op,
)
from mlpipeline.preprocessing.base import OpContext  # noqa: E402
from mlpipeline.preprocessing.pipeline import validate_config  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


def _baseline() -> tuple[PreprocessConfig, str]:
    return load_config_file(REPO / "ml/configs/preprocessing/baseline_sonar.yaml", PreprocessConfig)


class TestRegistry:
    def test_builtin_ops_registered(self):
        names = known_op_names()
        for expected in [
            "denoise_median", "denoise_nlm", "clahe",
            "normalize_intensity", "resize_letterbox", "optional_flip_waterfall",
        ]:
            assert expected in names

    def test_unknown_op_raises_with_known_list(self):
        with pytest.raises(KeyError, match="clahe"):
            from mlpipeline.preprocessing import get_op
            get_op("clahe_typo")

    def test_duplicate_registration_rejected(self):
        from mlpipeline.preprocessing.registry import PreprocessingRegistry

        r = PreprocessingRegistry()

        @r.register
        class A:
            name = "dup_test"

        with pytest.raises(ValueError, match="already registered"):
            r.register(A)


class TestEngineDeterminism:
    def test_deterministic_given_same_input_and_config(self):
        """Gate: same image + same config => bit-identical output.

        Per-op wall-clock timings are naturally variable and excluded from the
        equality check; everything else (ops, params, scale factors, warnings)
        must be identical.
        """
        cfg, h = _baseline()
        eng = PreprocessingEngine()
        img = make_sonar_like_image(seed=7)
        out1, arr1 = eng.run_with_array(img, cfg, h, "img_det")
        out2, arr2 = eng.run_with_array(img, cfg, h, "img_det")
        assert arr1.tobytes() == arr2.tobytes()
        d1, d2 = out1.model_dump(), out2.model_dump()
        for d in (d1, d2):
            for op in d["applied_ops"]:
                op.pop("duration_ms")  # wall-clock noise
        assert d1 == d2

    def test_records_applied_ops_params_and_timings(self):
        cfg, h = _baseline()
        eng = PreprocessingEngine()
        result, _ = eng.run_with_array(make_sonar_like_image(), cfg, h, "img_rec")
        applied = [a.op for a in result.applied_ops]
        # disabled ops (denoise_median etc.) must NOT appear
        assert applied == ["normalize_intensity", "clahe", "resize_letterbox"]
        norm = result.applied_ops[0]
        assert norm.params["low_percentile"] == 1.0
        assert norm.duration_ms >= 0.0

    def test_records_config_hash(self):
        cfg, h = _baseline()
        result, _ = PreprocessingEngine().run_with_array(
            make_sonar_like_image(), cfg, h, "img_hash"
        )
        assert result.config_hash == h and len(h) == 64

    def test_letterbox_records_scale_factors(self):
        cfg, h = _baseline()
        result, _ = PreprocessingEngine().run_with_array(
            make_sonar_like_image(width=200, height=100), cfg, h, "img_sf"
        )
        assert (result.width, result.height) == (640, 640)
        sf = result.scale_factors
        assert sf.scale_x == pytest.approx(640 / 200)
        assert sf.scale_y == pytest.approx(640 / 200)
        # pad centers the content
        content_w = 200 * sf.scale_x
        assert sf.pad_x == pytest.approx((640 - content_w) / 2, abs=1.0)

    def test_disabled_only_config_leaves_image_unresized(self):
        cfg = PreprocessConfig(
            name="noop",
            ops=[{"op": "denoise_median", "enabled": False, "params": {}}],
        )
        result, arr = PreprocessingEngine().run_with_array(
            make_sonar_like_image(width=64, height=32), cfg, "hash0", "img_noop"
        )
        assert arr.shape == (32, 64)
        assert result.applied_ops == []

    def test_unknown_op_fails_loudly_at_run(self):
        cfg = PreprocessConfig(name="bad", ops=[{"op": "nonexistent_op"}])
        with pytest.raises(PreprocessingError, match="unknown preprocessing op"):
            PreprocessingEngine().run(make_sonar_like_image(), cfg, "h", "img_bad")

    def test_bad_params_fail_with_op_name(self):
        cfg = PreprocessConfig(
            name="badp",
            ops=[{"op": "normalize_intensity", "params": {"low_percentile": 50, "high_percentile": 10}}],
        )
        with pytest.raises(PreprocessingError, match="normalize_intensity"):
            PreprocessingEngine().run(make_sonar_like_image(), cfg, "h", "img_badp")

    def test_validate_config_rejects_bad_params_before_run(self):
        cfg = PreprocessConfig(
            name="badv",
            ops=[{"op": "resize_letterbox", "params": {"target_size": [0, 0]}}],
        )
        with pytest.raises(PreprocessingError, match="resize_letterbox"):
            validate_config(cfg)


class TestOps:
    def test_normalize_output_range(self):
        from mlpipeline.preprocessing.ops.normalize import NormalizeIntensityOp

        img = make_sonar_like_image()
        out = NormalizeIntensityOp().apply(img, {"low_percentile": 1, "high_percentile": 99}, OpContext())
        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_normalize_does_not_mutate_input(self):
        from mlpipeline.preprocessing.ops.normalize import NormalizeIntensityOp

        img = make_sonar_like_image()
        before = img.copy()
        NormalizeIntensityOp().apply(img, {}, OpContext())
        assert np.array_equal(img, before)

    def test_clahe_enhances_contrast(self):
        from mlpipeline.preprocessing.ops.contrast import ClaheOp

        img = make_sonar_like_image()
        out = ClaheOp().apply(img, {"clip_limit": 2.0, "tile_grid_size": [8, 8]}, OpContext())
        assert out.dtype == np.float32
        assert out.std() >= (img.astype(np.float32) / 255.0).std() * 0.5

    def test_median_kernel_validation(self):
        from mlpipeline.preprocessing.ops.denoise import DenoiseMedianOp

        with pytest.raises(ValueError, match="kernel"):
            DenoiseMedianOp().apply(make_sonar_like_image(), {"kernel": 4}, OpContext())

    def test_flip_op(self):
        from mlpipeline.preprocessing.ops.orientation import OptionalFlipWaterfallOp

        img = make_sonar_like_image(width=10, height=4)
        ctx = OpContext()
        out = OptionalFlipWaterfallOp().apply(img, {"flip_horizontal": True}, ctx)
        assert np.array_equal(out, np.fliplr(img))
        ctx2 = OpContext()
        out2 = OptionalFlipWaterfallOp().apply(img, {}, ctx2)
        assert np.array_equal(out2, img)
        assert ctx2.warnings  # warns when enabled but no-op

    def test_scale_factor_coordinate_round_trip(self):
        """The ScaleFactors contract (Section 6.6): processed->source must invert."""
        sf = ScaleFactors(scale_x=0.5, scale_y=0.5, pad_x=10, pad_y=20)
        proc = ProcessedSonarImage(
            image_id="i", width=640, height=640, source_width=1000, source_height=2000,
            scale_factors=sf, config_hash="h",
        )
        for sx, sy in [(0, 0), (999, 1999)]:
            px, py = proc.to_processed_coords(sx, sy)
            rx, ry = proc.to_source_coords(px, py)
            assert rx == pytest.approx(sx, abs=1e-9)
            assert ry == pytest.approx(sy, abs=1e-9)
