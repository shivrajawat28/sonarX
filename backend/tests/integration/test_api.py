"""Steps 11-12 integration tests (TestClient + stub engine, temp DATA_ROOT).

Gate (Section 25, step 11): error envelope + path-traversal prevention.
Gate (Section 25, step 12): curl-equivalent upload -> run -> export works over HTTP.
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.services.inference_service import reset_inference_service


def _png_bytes(seed: int = 1, size: int = 96) -> bytes:
    import cv2

    rng = np.random.default_rng(seed)
    img = np.full((size, size), 60, dtype=np.uint8)
    img[size // 3 : size // 3 + 20, size // 3 : size // 3 + 20] = 240  # bright target
    img += (rng.random((size, size)) * 8).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """App with isolated DATA_ROOT and a registered stub model as active."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5173")
    Settings.model_rebuild()

    # register + activate a stub model (registry lives under the temp MODELS_DIR)
    (tmp_path / "models").mkdir(parents=True)
    registry_path = tmp_path / "models" / "registry.json"
    registry_path.write_text(_stub_registry_json())

    reset_inference_service()

    from backend.app.core.config import get_settings

    get_settings.cache_clear()
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        # ensure lifespan loaded model with the temp registry
        yield c, tmp_path


def _stub_registry_json() -> str:
    import json
    import hashlib
    from pathlib import Path

    pp = Path("ml/configs/preprocessing/baseline_sonar.yaml")
    pp_hash = hashlib.sha256(pp.read_bytes()).hexdigest()
    # stub checkpoint marker (weights file never decoded by StubDetector)
    return json.dumps({"models": [{
        "model_version": "stub-e2e-v1",
        "architecture_family": "stub",
        "framework": "numpy",
        "checkpoint_path": "models/weights/stub-e2e-v1/stub_checkpoint.json",
        "input_size": [640, 640],
        "class_map": {"0": "stub_class_a", "1": "stub_class_b"},
        "preprocess_config_ref": {"path": str(pp), "sha256": pp_hash},
        "status": "active",
        "notes": "INTEGRATION TEST fixture model",
        "created_at": "2026-09-09T00:00:00+00:00",
    }]})


class TestHealth:
    def test_health_reports_model_and_storage(self, client):
        c, tmp_path = client
        r = c.get("/api/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert body["storage_ok"] is True
        assert body["model"]["loaded"] is True
        assert body["model"]["version"] == "stub-e2e-v1"
        assert "x-request-id" in r.headers

    def test_health_degraded_without_model(self, tmp_path, monkeypatch):
        """Section 11.4: app starts in degraded mode when no model registered."""
        monkeypatch.setenv("DATA_ROOT", str(tmp_path / "d2"))
        monkeypatch.setenv("MODELS_DIR", str(tmp_path / "m2"))
        (tmp_path / "m2").mkdir(parents=True)
        (tmp_path / "m2" / "registry.json").write_text('{"models": []}')
        Settings.model_rebuild()
        reset_inference_service()

        from backend.app.core.config import get_settings

        get_settings.cache_clear()
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/api/v1/health")
            assert r.status_code == 200
            assert r.json()["model"]["loaded"] is False
            # detection endpoints must 503 MODEL_UNAVAILABLE, not 500
            r2 = c.post("/api/v1/detections/run", json={"image_id": "img_x"})
            assert r2.status_code == 503
            assert r2.json()["error"]["code"] == "MODEL_UNAVAILABLE"


class TestUploadDetectExport:
    def test_full_http_flow(self, client):
        """Gate: upload -> run -> save -> history -> export (HTTP only)."""
        c, tmp_path = client

        # 1. upload
        r = c.post("/api/v1/uploads/image",
                   files={"file": ("sonar_sample.png", io.BytesIO(_png_bytes()), "image/png")})
        assert r.status_code == 201, r.text
        image_id = r.json()["image_id"]
        assert r.json()["sha256"]

        # 2. reject garbage upload with proper envelope
        r_bad = c.post("/api/v1/uploads/image",
                       files={"file": ("fake.png", io.BytesIO(b"not a png at all"), "image/png")})
        assert r_bad.status_code == 415  # magic-byte mismatch -> UNSUPPORTED_FILE_TYPE
        assert r_bad.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"

        r_ext = c.post("/api/v1/uploads/image",
                       files={"file": ("evil.exe", io.BytesIO(b"MZ..."), "application/x-msdownload")})
        assert r_ext.status_code == 415
        assert r_ext.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"

        # 3. run detection (sync)
        r_run = c.post("/api/v1/detections/run", json={"image_id": image_id, "save": True})
        assert r_run.status_code == 201, r_run.text
        result = r_run.json()
        assert result["model_version"] == "stub-e2e-v1"
        assert len(result["preprocess_config_hash"]) == 64
        assert len(result["detections"]) > 0
        det = result["detections"][0]
        assert det["latitude"] is None and det["geo_status"] == "unavailable"  # no fabrication
        assert det["filtering_status"] in ("accepted", "flagged", "rejected")

        # 4. history + single detection
        r_hist = c.get("/api/v1/detections")
        assert r_hist.status_code == 200 and r_hist.json()["total"] >= 1
        r_det = c.get(f"/api/v1/detections/{det['detection_id']}")
        assert r_det.status_code == 200

        # 5. analyst override
        r_patch = c.patch(f"/api/v1/detections/{det['detection_id']}",
                          json={"status": "accepted", "note": "visually verified"})
        assert r_patch.status_code == 200
        assert r_patch.json()["analyst_overridden"] is True

        # 6. exports (stable CSV columns)
        r_csv = c.get("/api/v1/exports/detections.csv")
        assert r_csv.status_code == 200
        header = r_csv.text.splitlines()[0]
        assert header.startswith("detection_id,run_id,image_id,class_name")
        assert "model_confidence" in header and "geo_status" in header
        assert str(det["detection_id"]) in r_csv.text

        r_json = c.get("/api/v1/exports/detections.json")
        assert r_json.status_code == 200
        assert r_json.json()["count"] >= 1

        # 7. run detail
        r_runget = c.get(f"/api/v1/detections/runs/{result['detection_run_id']}")
        assert r_runget.status_code == 200
        assert r_runget.json()["kind"] == "single_image"

    def test_processed_preview_endpoints(self, client):
        c, _ = client
        r = c.post("/api/v1/uploads/image",
                   files={"file": ("p.png", io.BytesIO(_png_bytes(9)), "image/png")})
        image_id = r.json()["image_id"]

        r_prev = c.post("/api/v1/previews/preprocess", json={"image_id": image_id})
        assert r_prev.status_code == 200, r_prev.text
        assert len(r_prev.json()["config_hash"]) == 64
        assert r_prev.json()["applied_ops"]

        r_img = c.get(f"/api/v1/images/{image_id}/processed")
        assert r_img.status_code == 200
        assert r_img.headers["content-type"].startswith("image/")

    def test_unknown_ids_404_envelope(self, client):
        c, _ = client
        for url in ("/api/v1/detections/det_missing", "/api/v1/images/img_missing",
                    "/api/v1/surveys/srv_missing", "/api/v1/jobs/job_missing"):
            r = c.get(url)
            assert r.status_code == 404, url
            assert r.json()["error"]["code"] == "NOT_FOUND"


class TestSurveyBatch:
    def _zip_with_nav(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for i in range(3):
                zf.writestr(f"tile_{i}.png", _png_bytes(seed=i + 20))
            zf.writestr("nav.csv",
                        "timestamp,latitude,longitude\n"
                        "2026-09-01T10:00:00Z,19.000,72.000\n"
                        "2026-09-01T10:00:09Z,19.010,72.000\n")
        return buf.getvalue()

    def test_survey_upload_batch_job_flow(self, client):
        c, _ = client
        r = c.post("/api/v1/uploads/survey",
                   files={"file": ("survey.zip", io.BytesIO(self._zip_with_nav()), "application/zip")},
                   data={"name": "test pass"})
        assert r.status_code == 201, r.text
        survey = r.json()
        assert survey["image_count"] == 3
        assert survey["navigation_status"] == "present"

        # trigger async batch
        r_run = c.post(f"/api/v1/surveys/{survey['survey_id']}/run", json={"save": True})
        assert r_run.status_code == 202, r_run.text
        job_id = r_run.json()["job_id"]

        # poll until done (in-process BackgroundTasks run on response completion)
        import time

        job = None
        for _ in range(50):
            r_job = c.get(f"/api/v1/jobs/{job_id}")
            job = r_job.json()
            if job["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.1)
        assert job["status"] == "succeeded", job
        assert job["result_ref"]["n_detections"] > 0

        # detections persisted and browsable via history filter
        r_hist = c.get("/api/v1/detections", params={"survey_id": survey["survey_id"]})
        assert r_hist.status_code == 200
        assert r_hist.json()["total"] > 0

    def test_survey_upload_rejects_path_traversal(self, client):
        c, _ = client
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../../../evil.png", _png_bytes())
        r = c.post("/api/v1/uploads/survey",
                   files={"file": ("evil.zip", io.BytesIO(buf.getvalue()), "application/zip")})
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "CORRUPT_SONAR_DATA"


class TestModelsEndpoints:
    def test_models_list_and_metrics(self, client):
        c, tmp_path = client
        r = c.get("/api/v1/models")
        assert r.status_code == 200
        items = r.json()["items"]
        assert items and items[0]["model_version"] == "stub-e2e-v1"

        r2 = c.get("/api/v1/models/stub-e2e-v1")
        assert r2.status_code == 200
        assert r2.json()["class_map"] == {"0": "stub_class_a", "1": "stub_class_b"}

        # no evaluation runs yet -> honest 404, never fabricated metrics
        r3 = c.get("/api/v1/models/stub-e2e-v1/metrics")
        assert r3.status_code == 404
        assert "no evaluation runs" in r3.json()["error"]["message"]

    def test_unknown_model_404(self, client):
        c, _ = client
        assert c.get("/api/v1/models/never-v9").status_code == 404


class TestReports:
    def test_report_job_flow(self, client):
        c, _ = client
        r_up = c.post("/api/v1/uploads/image",
                      files={"file": ("r.png", io.BytesIO(_png_bytes(7)), "image/png")})
        image_id = r_up.json()["image_id"]
        c.post("/api/v1/detections/run", json={"image_id": image_id, "save": True})

        r = c.post("/api/v1/reports", json={"run_id": None, "survey_id": None})
        assert r.status_code in (400, 422)  # requires a subject — validation

        # use a survey-less run: create report for the saved run
        r_hist = c.get("/api/v1/detections/runs", follow_redirects=False)
        # find the run id via detections
        dets = c.get("/api/v1/detections").json()["items"]
        run_id = dets[0]["run_id"]
        r2 = c.post("/api/v1/reports", json={"run_id": run_id})
        assert r2.status_code == 202, r2.text
        job_id = r2.json()["job_id"]

        import time

        for _ in range(50):
            job = c.get(f"/api/v1/jobs/{job_id}").json()
            if job["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.1)
        assert job["status"] == "succeeded", job
        report_id = job["result_ref"]["report_id"]

        r_rep = c.get(f"/api/v1/reports/{report_id}")
        assert r_rep.status_code == 200
        assert "Coordinates are present ONLY" in r_rep.text  # honesty notice present
