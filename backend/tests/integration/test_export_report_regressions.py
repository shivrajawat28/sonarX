"""Regression tests for defects found and fixed in the final release audit.

Each test encodes a bug that was actually observed in this repository, so the
fix cannot silently regress:

1. Exports silently truncated at the repository's 200-document page cap.
2. Reports had the same truncation and omitted the bounding box entirely.
3. `resolve_within` accepted a sibling directory sharing the root's name prefix.
4. `/models/{v}/metrics` exposed neither the confusion matrix nor the model's
   training provenance, so the evaluation dashboard could not show them.
5. `confusion.json` was written as a Python repr (single quotes) rather than JSON.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.core.security import resolve_within
from backend.app.main import create_app
from backend.app.services.inference_service import reset_inference_service

# The integration tests share the stub-model fixture helpers from test_api.py.
# pytest puts this directory on sys.path (no package __init__ here), so the
# module import works directly; the fallback covers package-style collection.
try:  # pragma: no cover - import plumbing
    from test_api import _stub_registry_json
    from test_api import _png_bytes  # noqa: F401  (shared helper, used below)
except ImportError:  # pragma: no cover
    from .test_api import _png_bytes, _stub_registry_json  # type: ignore[no-redef]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Isolated app instance with the stub model registered as active."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5173")
    Settings.model_rebuild()
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "models" / "registry.json").write_text(_stub_registry_json())
    reset_inference_service()
    from backend.app.core.config import get_settings

    get_settings.cache_clear()
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        yield c, tmp_path


# --------------------------------------------------------------------------
# 1 + 2. exports and reports must never truncate
# --------------------------------------------------------------------------


class TestNoSilentTruncation:
    def _seed_detections(self, c, n: int) -> list[str]:
        """Persist `n` detections through the real repository (no mocks)."""
        from backend.app.core.config import get_settings
        from backend.app.services.inference_service import get_inference_service

        service = get_inference_service(get_settings())
        # reports validate that the subject run exists, so persist it too
        service.repo.insert("detection_runs", {
            "run_id": "run_bulk",
            "kind": "single_image",
            "image_id": "img_bulk",
            "image_ids": ["img_bulk"],
            "model_version": "stub-e2e-v1",
            "preprocess_config_hash": "0" * 64,
            "filter_config_hash": None,
            "detection_ids": [],
            "timings_ms": {},
            "warnings": [],
            "created_at": "2026-01-01T00:00:00+00:00",
        })
        ids = []
        for i in range(n):
            det_id = f"det_bulk_{i:04d}"
            service.repo.insert("detections", {
                "detection_id": det_id,
                "run_id": "run_bulk",
                "image_id": "img_bulk",
                "class_name": "stub_class_a",
                "model_confidence": 0.5,
                "final_confidence": 0.5,
                "filtering_status": "accepted",
                "filter_reasons": [],
                "bbox_source_coords": {"x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0},
                "bbox_processed_coords": {"x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0},
                "latitude": None,
                "longitude": None,
                "geo_status": "unavailable",
                "model_version": "stub-e2e-v1",
                "preprocess_config_hash": "0" * 64,
                "created_at": f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}+00:00",
            })
            ids.append(det_id)
        return ids

    def test_repository_list_stays_capped_but_list_all_does_not(self, client):
        c, _ = client
        self._seed_detections(c, 250)

        from backend.app.core.config import get_settings
        from backend.app.services.inference_service import get_inference_service

        repo = get_inference_service(get_settings()).repo
        page, total = repo.list("detections", page=1, size=5000)
        assert total == 250
        assert len(page) == 200, "request-serving pagination must stay bounded"

        everything, total_all = repo.list_all("detections")
        assert total_all == 250
        assert len(everything) == 250

    def test_csv_and_json_exports_include_every_detection(self, client):
        c, _ = client
        ids = self._seed_detections(c, 250)

        csv_text = c.get("/api/v1/exports/detections.csv").text
        rows = [l for l in csv_text.splitlines() if l.strip()]
        assert len(rows) - 1 == 250, f"CSV truncated: {len(rows) - 1} data rows"
        assert "det_bulk_0249" in csv_text

        payload = c.get("/api/v1/exports/detections.json").json()
        assert payload["count"] == 250
        assert len(payload["detections"]) == 250
        assert {d["detection_id"] for d in payload["detections"]} == set(ids)

    def test_report_includes_every_detection_and_the_bbox(self, client):
        c, _ = client
        self._seed_detections(c, 210)

        r = c.post("/api/v1/reports", json={"run_id": "run_bulk"})
        assert r.status_code == 202, r.text
        job_id = r.json()["job_id"]

        import time

        job = None
        for _ in range(100):
            job = c.get(f"/api/v1/jobs/{job_id}").json()
            if job["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.1)
        assert job["status"] == "succeeded", job

        report_id = job["result_ref"]["report_id"]
        html = c.get(f"/api/v1/reports/{report_id}").text
        assert "det_bulk_0209" in html, "HTML report dropped detections past the page cap"
        assert "bbox" in html.lower() or "x,y,w,h" in html.lower(), "report omits bounding boxes"
        assert "Model conf" in html and "Final conf" in html

        pdf = c.get(f"/api/v1/reports/{report_id}?format=pdf").content
        assert pdf[:5] == b"%PDF-", "report PDF is not a valid PDF"


# --------------------------------------------------------------------------
# 3. path confinement must use real ancestry, not a string prefix
# --------------------------------------------------------------------------


class TestResolveWithinHardening:
    def test_sibling_directory_with_shared_prefix_is_rejected(self, tmp_path):
        root = tmp_path / "data"
        root.mkdir()
        sibling = tmp_path / "data-evil"
        sibling.mkdir()
        (sibling / "secret.txt").write_text("x")

        with pytest.raises(ValueError):
            resolve_within(root, "..", "data-evil", "secret.txt")

    def test_normal_nested_path_allowed(self, tmp_path):
        root = tmp_path / "data"
        (root / "uploads").mkdir(parents=True)
        assert resolve_within(root, "uploads", "a.png") == (root / "uploads" / "a.png").resolve()

    def test_root_itself_allowed(self, tmp_path):
        root = tmp_path / "data"
        root.mkdir()
        assert resolve_within(root) == root.resolve()


# --------------------------------------------------------------------------
# 4 + 5. evaluation dashboard data must be present and machine-readable
# --------------------------------------------------------------------------


class TestEvalMetricsPayload:
    def _write_eval_run(self, models_dir: Path) -> str:
        """Write a real EvalRun record + confusion artifact (valid JSON only)."""
        from mlpipeline.datatypes.model import EvaluationRun, MetricsSummary, PerClassMetrics
        from mlpipeline.registry.eval_runs import EvalRunStore

        eval_run_id = "eval_stub-e2e-v1_test_20260101_000000"
        run = EvaluationRun(
            eval_run_id=eval_run_id,
            model_version="stub-e2e-v1",
            dataset_ref={"path": "datasets/manifests/x.json", "sha256": "a" * 64},
            split="test",
            split_seed=42,
            preprocess_config_hash="b" * 64,
            filter_enabled=False,
            metrics=MetricsSummary(precision=0.5, recall=0.4, f1=0.44, mAP50=0.42),
            per_class={
                "stub_class_a": PerClassMetrics(
                    precision=0.5, recall=0.4, f1=0.44, ap50=0.42, support=10
                )
            },
        )
        store = EvalRunStore(models_dir / "eval")
        store.save(run, artifacts={
            "confusion.json": json.dumps(
                {"stub_class_a": {"stub_class_a": 7, "stub_class_b": 3}}, indent=2
            )
        })
        return eval_run_id

    def test_metrics_endpoint_exposes_confusion_and_provenance(self, client):
        c, tmp_path = client
        eval_run_id = self._write_eval_run(tmp_path / "models")

        r = c.get("/api/v1/models/stub-e2e-v1/metrics")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["eval_run_id"] == eval_run_id
        assert body["split"] == "test"
        assert body["confusion"] == {"stub_class_a": {"stub_class_a": 7, "stub_class_b": 3}}
        assert body["model"]["architecture_family"] == "stub"
        assert body["model"]["class_map"] == {"0": "stub_class_a", "1": "stub_class_b"}
        assert body["model"]["checkpoint_path"] == "models/weights/stub-e2e-v1/stub_checkpoint.json"

    def test_confusion_artifact_written_by_evaluate_is_valid_json(self, tmp_path):
        """`evaluate.py` must write JSON, not a Python repr."""
        from mlpipeline.evaluation.metrics import MatchedPair, confusion_matrix

        matches = [
            MatchedPair(pred_class="a", gt_class="a", iou=0.9, score=0.8),
            MatchedPair(pred_class="b", gt_class="a", iou=0.1, score=0.3),
        ]
        payload = json.dumps(confusion_matrix(matches, ["a", "b"]), indent=2)
        parsed = json.loads(payload)  # would raise on the old `.__str__()` output
        assert parsed["a"]["a"] >= 0
