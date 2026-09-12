"""Step 8 tests: post-processing + false-positive filtering."""
import numpy as np
import pytest

from mlpipeline.config.schemas import DetectionConfig, FilterConfig
from mlpipeline.detection.base import ModelMeta
from mlpipeline.datatypes.detection import BBox, Detection, RawDetection
from mlpipeline.datatypes.image import ProcessedSonarImage, ScaleFactors
from mlpipeline.filtering import FilterContext, FilterPipeline
from mlpipeline.filtering.rule_registry import known_rule_names
from mlpipeline.filtering.scorer import score_detection, status_for
from mlpipeline.filtering.rules.validator_stub import SecondaryValidatorStub
from mlpipeline.postprocessing import PostProcessor, iou, nms_numpy


def _processed(w=640, h=640):
    return ProcessedSonarImage(
        image_id="img_p", width=w, height=h, source_width=1000, source_height=1000,
        scale_factors=ScaleFactors(scale_x=0.64, scale_y=0.64, pad_x=0, pad_y=0), config_hash="ph",
    )


def _meta():
    return ModelMeta(model_version="stub-test-v0", architecture_family="stub",
                     class_map={0: "alpha", 1: "beta"}, input_size=[640, 640], framework="numpy")


class TestNMS:
    def test_suppresses_overlapping_keeps_best(self):
        boxes = [BBox(x=0, y=0, w=10, h=10), BBox(x=1, y=1, w=10, h=10), BBox(x=50, y=50, w=10, h=10)]
        scores = [0.9, 0.8, 0.7]
        keep = nms_numpy(boxes, scores, 0.5)
        assert keep == [0, 2]

    def test_identical_boxes_keep_one(self):
        boxes = [BBox(x=5, y=5, w=10, h=10), BBox(x=5, y=5, w=10, h=10)]
        assert nms_numpy(boxes, [0.6, 0.9], 0.5) == [1]

    def test_iou_identical_is_one(self):
        assert iou(BBox(x=0, y=0, w=10, h=10), BBox(x=0, y=0, w=10, h=10)) == 1.0
        assert iou(BBox(x=0, y=0, w=10, h=10), BBox(x=100, y=100, w=10, h=10)) == 0.0


class TestPostProcessor:
    def _raw(self):
        return [
            RawDetection(class_id=0, score=0.9, box=BBox(x=10, y=10, w=50, h=50)),
            RawDetection(class_id=0, score=0.6, box=BBox(x=12, y=12, w=50, h=50)),  # NMS'd
            RawDetection(class_id=1, score=0.2, box=BBox(x=200, y=200, w=40, h=40)),  # thresholded
            RawDetection(class_id=0, score=0.8, box=BBox(x=300, y=300, w=60, h=60)),
        ]

    def test_threshold_applied_once_and_recorded(self):
        res = PostProcessor().process(self._raw(), _processed(), _meta(), DetectionConfig(confidence_threshold=0.3))
        assert res.applied_confidence_threshold == 0.3
        kept_scores = [d.model_confidence for d in res.detections]
        assert all(s >= 0.3 for s in kept_scores)
        assert res.n_dropped_by_threshold == 1

    def test_class_names_resolved_from_map(self):
        res = PostProcessor().process(self._raw(), _processed(), _meta(), DetectionConfig())
        assert {d.class_name for d in res.detections} <= {"alpha", "beta"}
        assert not any(d.class_name.startswith("0") for d in res.detections)

    def test_unknown_class_id_refuses_to_guess(self):
        meta = _meta()
        meta.class_map = {0: "alpha"}  # id 1 missing
        with pytest.raises(Exception, match="class_id"):
            PostProcessor().process(self._raw(), _processed(), meta, DetectionConfig(confidence_threshold=0.1))

    def test_boxes_rescaled_to_source_coords(self):
        res = PostProcessor().process(self._raw(), _processed(), _meta(), DetectionConfig())
        d = res.detections[0]
        assert d.bbox_source_coords.x == pytest.approx(10 / 0.64, abs=0.6)
        assert d.bbox_processed_coords == BBox(x=10, y=10, w=50, h=50)

    def test_model_confidence_untouched(self):
        res = PostProcessor().process(self._raw(), _processed(), _meta(), DetectionConfig())
        scores = sorted(d.model_confidence for d in res.detections)
        assert scores[-1] == pytest.approx(0.9)
        assert all(d.final_confidence == d.model_confidence for d in res.detections)  # pre-filter


class TestFilterRules:
    def _det(self, x=100, y=100, w=50, h=50, cls="alpha", conf=0.8):
        return Detection(
            image_id="img", class_name=cls, model_confidence=conf,
            bbox_source_coords=BBox(x=x, y=y, w=w, h=h),
            model_version="m", preprocess_config_hash="p",
        )

    def _ctx(self, w=1000, h=1000, pixels=None):
        lo, hi = (0.2, 0.8) if pixels is None else (None, None)
        return FilterContext(image_width=w, image_height=h, pixels=pixels,
                             image_intensity_p10=lo, image_intensity_p90=hi)

    def test_known_rules_registered(self):
        names = known_rule_names()
        for expected in ["min_size", "aspect_ratio", "edge_clip", "intensity_outlier",
                         "shadow_ratio", "class_specific", "secondary_validator"]:
            assert expected in names

    def test_min_size_triggers_on_tiny_box(self):
        from mlpipeline.filtering.rules.min_size import MinSizeRule

        v = MinSizeRule().evaluate(self._det(w=2, h=2), self._ctx(), {"min_area_frac": 0.0004})
        assert v.triggered and "min_size" in v.reason

    def test_aspect_ratio_triggers_on_extreme(self):
        from mlpipeline.filtering.rules.aspect_ratio import AspectRatioRule

        v = AspectRatioRule().evaluate(self._det(w=100, h=2), self._ctx(), {"max_ratio": 8.0})
        assert v.triggered

    def test_edge_clip_triggers_on_border(self):
        from mlpipeline.filtering.rules.edge_clip import EdgeClipRule

        v = EdgeClipRule().evaluate(self._det(x=0, y=100), self._ctx(), {})
        assert v.triggered
        v2 = EdgeClipRule().evaluate(self._det(x=100, y=100), self._ctx(), {})
        assert not v2.triggered

    def test_intensity_outlier_skips_without_pixels(self):
        from mlpipeline.filtering.rules.intensity_outlier import IntensityOutlierRule

        v = IntensityOutlierRule().evaluate(self._det(), self._ctx(), {})
        assert not v.triggered  # cannot evaluate -> no penalty (never fabricates evidence)

    def test_intensity_outlier_flags_flat_region(self):
        from mlpipeline.filtering.rules.intensity_outlier import IntensityOutlierRule

        img = np.full((1000, 1000), 0.5, dtype=np.float32)
        ctx = self._ctx(pixels=img)
        lo, hi = np.percentile(img, [10, 90])
        ctx.image_intensity_p10, ctx.image_intensity_p90 = float(lo), float(hi)
        v = IntensityOutlierRule().evaluate(self._det(), ctx, {"min_contrast": 0.02})
        assert v.triggered  # region == background -> texture noise

    def test_secondary_validator_stub_inert_without_model(self):
        v = SecondaryValidatorStub().evaluate(self._det(), self._ctx(), {})
        assert not v.triggered  # stub is a no-op at MVP


class TestFilterPipeline:
    def _pipeline(self, cfg: FilterConfig | None = None):
        cfg = cfg or FilterConfig(
            name="t",
            enabled_rules=[
                {"rule": "min_size", "params": {"min_area_frac": 0.0004}},
                {"rule": "edge_clip", "params": {}},
            ],
        )
        return FilterPipeline(cfg, config_hash="fh123")

    def test_annotates_never_deletes(self):
        pipe = self._pipeline()
        dets = [
            Detection(image_id="i", class_name="alpha", model_confidence=0.9,
                      bbox_source_coords=BBox(x=100, y=100, w=100, h=100),
                      model_version="m", preprocess_config_hash="p"),
            Detection(image_id="i", class_name="alpha", model_confidence=0.9,
                      bbox_source_coords=BBox(x=1, y=1, w=1, h=1),
                      model_version="m", preprocess_config_hash="p"),
        ]
        res = pipe.apply(dets, FilterContext(image_width=1000, image_height=1000))
        assert len(res.detections) == 2  # NOTHING deleted
        statuses = {d.filtering_status for d in res.detections}
        assert statuses <= {"accepted", "flagged", "rejected"}
        assert res.detections[0].model_confidence == 0.9  # raw preserved

    def test_status_policy_thresholds(self):
        assert status_for(0.9, 0.6, 0.35) == "accepted"
        assert status_for(0.5, 0.6, 0.35) == "flagged"
        assert status_for(0.2, 0.6, 0.35) == "rejected"

    def test_final_confidence_never_negative(self):
        pipe = self._pipeline()
        d = Detection(image_id="i", class_name="alpha", model_confidence=0.1,
                      bbox_source_coords=BBox(x=1, y=1, w=1, h=1),
                      model_version="m", preprocess_config_hash="p")
        res = pipe.apply([d], FilterContext(image_width=1000, image_height=1000))
        assert res.detections[0].final_confidence == 0.0
        assert res.detections[0].filtering_status == "rejected"
        assert res.detections[0].filter_reasons  # explainable

    def test_filter_config_hash_recorded(self):
        pipe = self._pipeline()
        d = Detection(image_id="i", class_name="alpha", model_confidence=0.9,
                      bbox_source_coords=BBox(x=100, y=100, w=100, h=100),
                      model_version="m", preprocess_config_hash="p")
        res = pipe.apply([d], FilterContext(image_width=1000, image_height=1000))
        assert res.detections[0].filter_config_hash == "fh123"

    def test_bundled_rules_yaml_loads_and_applies(self):
        from pathlib import Path
        from mlpipeline.config import load_config_file

        cfg, h = load_config_file(Path("ml/configs/filtering/rules.yaml"), FilterConfig)
        pipe = FilterPipeline(cfg, h)
        res = pipe.apply(
            [Detection(image_id="i", class_name="alpha", model_confidence=0.9,
                       bbox_source_coords=BBox(x=100, y=100, w=200, h=200),
                       model_version="m", preprocess_config_hash="p")],
            FilterContext(image_width=1000, image_height=1000),
        )
        assert res.detections[0].filtering_status == "accepted"
        assert res.n_accepted == 1
