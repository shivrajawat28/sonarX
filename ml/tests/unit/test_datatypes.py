"""Step 2 acceptance tests: canonical datatypes round-trip serialization.

Gate (Section 25, step 2): datatype serialization tests pass.
"""
import json

import pytest

from mlpipeline.datatypes import (
    BBox,
    Detection,
    DetectionRun,
    EvaluationRun,
    GeoStatus,
    FilterStatus,
    InferenceResult,
    ModelVersion,
    NavigationSample,
    NavigationTrack,
    ProcessedSonarImage,
    RawDetection,
    ScaleFactors,
    SonarImage,
    Survey,
)
from mlpipeline.datatypes.image import AppliedOp


def roundtrip(model):
    """model -> JSON -> model must preserve data.

    Deterministic identity for default_factory fields: pin the generated id/timestamp
    of the ORIGINAL onto the round-tripped copy (defaults generate fresh values,
    which is correct behavior for new records, not for round-trips).
    """
    rt = type(model).model_validate(json.loads(model.model_dump_json()))
    if hasattr(model, "detection_id"):
        rt = rt.model_copy(update={"detection_id": model.detection_id, "created_at": model.created_at})
    if hasattr(model, "eval_run_id"):
        rt = rt.model_copy(update={"eval_run_id": model.eval_run_id, "timestamp": model.timestamp})
    return rt


class TestBBox:
    def test_properties(self):
        b = BBox(x=10, y=20, w=30, h=5)
        assert b.x2 == 40 and b.y2 == 25
        assert b.area == 150
        assert b.aspect_ratio == 6.0

    def test_zero_area_aspect_is_inf(self):
        assert BBox(x=0, y=0, w=5, h=0).aspect_ratio == float("inf")

    def test_clipping(self):
        b = BBox(x=-5, y=-5, w=20, h=200).clipped(100, 100)
        assert (b.x, b.y, b.w, b.h) == (0, 0, 15, 100)

    def test_roundtrip(self):
        assert roundtrip(BBox(x=1.5, y=2, w=3, h=4)) == BBox(x=1.5, y=2, w=3, h=4)


class TestScaleFactorsRoundTrip:
    def test_letterbox_round_trip(self):
        """source -> processed -> source must be identity (Step 4 will reuse this)."""
        sf = ScaleFactors(scale_x=0.5, scale_y=0.5, pad_x=10, pad_y=20)
        p = ProcessedSonarImage(
            image_id="img_x",
            width=640,
            height=640,
            source_width=1000,
            source_height=2000,
            scale_factors=sf,
            applied_ops=[AppliedOp(op="clahe", params={"clip": 2.0}, duration_ms=1.2)],
            config_hash="abc123",
        )
        for sx, sy in [(0, 0), (500, 1000), (999, 1999), (123.4, 567.8)]:
            px, py = p.to_processed_coords(sx, sy)
            rx, ry = p.to_source_coords(px, py)
            assert rx == pytest.approx(sx, abs=1e-9)
            assert ry == pytest.approx(sy, abs=1e-9)

    def test_processed_image_roundtrip(self):
        p = ProcessedSonarImage(
            image_id="img_x", width=640, height=640, source_width=800, source_height=600,
            scale_factors=ScaleFactors(scale_x=0.8, scale_y=1.0666), config_hash="h",
        )
        assert roundtrip(p) == p


class TestDetections:
    def _detection(self) -> Detection:
        return Detection(
            image_id="img_1",
            class_name="pipe",
            model_confidence=0.91,
            final_confidence=0.84,
            filtering_status="flagged",
            filter_reasons=["aspect_ratio: 9.2 > 8.0 (penalty 0.07)"],
            bbox_source_coords=BBox(x=120, y=80, w=300, h=240),
            bbox_processed_coords=BBox(x=96, y=64, w=240, h=192),
            latitude=None,
            longitude=None,
            geo_status="unavailable",
            model_version="yolo-n-sonar-v0.3.1",
            preprocess_config_hash="p_hash",
            filter_config_hash="f_hash",
        )

    def test_full_roundtrip(self):
        d = self._detection()
        rt = roundtrip(d)
        assert rt.model_dump() == d.model_dump()  # dump-equality: ids pinned in roundtrip()

    def test_model_vs_final_confidence_distinct(self):
        """ADR-006: raw model confidence must be preserved untouched."""
        d = self._detection()
        assert d.model_confidence == 0.91 and d.final_confidence == 0.84
        assert d.filtering_status == "flagged" and d.filter_reasons

    def test_rejected_detection_is_kept_not_deleted(self):
        d = self._detection().model_copy(
            update={"filtering_status": "rejected", "final_confidence": 0.10}
        )
        rt = roundtrip(d)
        assert rt.filtering_status == "rejected"
        assert rt.model_confidence == 0.91  # raw score survives
        assert rt.bbox_source_coords == d.bbox_source_coords  # nothing vanished

    def test_geo_null_propagation(self):
        """No fabrication: missing coords stay null through serialization."""
        d = self._detection()
        rt = roundtrip(d)
        assert rt.latitude is None and rt.longitude is None
        assert rt.geo_status == "unavailable"
        assert not rt.has_coordinates()

    def test_raw_detection_separate_from_canonical(self):
        r = RawDetection(class_id=2, score=0.55, box=BBox(x=1, y=2, w=3, h=4))
        assert roundtrip(r) == r
        assert r.class_id == 2  # bare id — class names come later, from class_map

    def test_confidence_bounds_enforced(self):
        with pytest.raises(Exception):
            RawDetection(class_id=0, score=1.5, box=BBox(x=0, y=0, w=1, h=1))


class TestNavigationSurvey:
    def test_navigation_track_roundtrip_preserves_raw_rows(self):
        t = NavigationTrack(
            format="generic_csv",
            source_ref="data/uploads/nav.csv",
            samples=[
                NavigationSample(index=0, timestamp="2026-09-01T10:00:00+00:00", ping_index=0,
                                 latitude=19.0760, longitude=72.8777, heading_deg=90.0,
                                 raw_row={"ts": "10:00:00", "lat": "19.0760", "lon": "72.8777"}),
                NavigationSample(index=1, timestamp="2026-09-01T10:00:01+00:00", ping_index=1,
                                 latitude=19.0761, longitude=72.8778, raw_row={"ts": "10:00:01"}),
            ],
        )
        rt = roundtrip(t)
        assert rt == t
        assert rt.samples[0].raw_row["lat"] == "19.0760"  # verbatim evidence preserved

    def test_coordinate_bounds_enforced(self):
        with pytest.raises(Exception):
            NavigationSample(index=0, latitude=95.0, longitude=0.0)

    def test_survey_roundtrip(self):
        s = Survey(name="test pass", image_count=2, images=["img_a", "img_b"])
        assert roundtrip(s) == s


class TestResultsAndRegistry:
    def test_inference_result_roundtrip(self):
        r = InferenceResult(
            image_id="img_1",
            model_version="m",
            preprocess_config_hash="p",
            filter_config_hash="f",
            warnings=["navigation metadata missing — coordinates unavailable"],
            timings_ms={"preprocess_ms": 11, "inference_ms": 480},
        )
        rt = roundtrip(r)
        assert rt == r
        assert rt.warnings == [r.warnings[0]]

    def test_detection_run_roundtrip(self):
        r = DetectionRun(kind="survey_batch", survey_id="srv_1", model_version="m",
                         preprocess_config_hash="p", image_ids=["img_a", "img_b"])
        assert roundtrip(r) == r

    def test_model_version_roundtrip(self):
        m = ModelVersion(
            model_version="yolo-n-sonar-v0.3.1",
            architecture_family="yolo",
            checkpoint_path="models/weights/x/best.pt",
            class_map={"0": "shipwreck", "1": "pipe"},
            preprocess_config_ref={"path": "ml/configs/preprocessing/baseline_sonar.yaml", "sha256": "x"},
            status="active",
        )
        rt = roundtrip(m)
        assert rt == m
        assert rt.class_map["1"] == "pipe"

    def test_evaluation_run_roundtrip(self):
        e = EvaluationRun(
            model_version="m",
            dataset_ref={"path": "datasets/manifests/sonar-v0.2.json", "sha256": "d"},
            split="test",
            split_seed=42,
            preprocess_config_hash="p",
            filter_enabled=True,
            per_class={"pipe": {"ap50": 0.6, "precision": 0.7, "recall": 0.5}},
        )
        rt = roundtrip(e)
        assert rt == e
        assert rt.per_class["pipe"].ap50 == 0.6
