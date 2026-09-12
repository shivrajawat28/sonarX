"""Seed demo data — CLEARLY LABELLED DEMO FIXTURES (Section 4 rule).

Creates:
- A stub model registered as active (StubDetector; synthetic detections only)
- A demo survey of synthetic sonar-like tiles + a synthetic nav CSV track

Rules honoured here:
- The nav track is SYNTHETIC and marked as such in survey metadata; it exists
  to exercise the geolocation plumbing, never to present real locations.
- No metrics/evaluations are fabricated. The Models page will honestly show
  "no evaluation runs" until a real evaluation is performed.
- The seeded images are synthetic patterns, not real sonar data.

Usage: make seed   (or .venv/bin/python scripts/seed_demo.py)
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ml"))
sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.persistence.file_repository import get_file_repository  # noqa: E402
from backend.app.services.storage_service import get_storage_service  # noqa: E402
from backend.app.services.inference_service import reset_inference_service  # noqa: E402


def _demo_tile(seed: int, size: int = 128) -> bytes:
    """Synthetic sonar-like pattern (textured background + bright blob)."""
    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)
    img = np.full((size, size), 60, dtype=np.uint8)
    # seafloor-like texture
    img += (rng.random((size, size)) * 24).astype(np.uint8)
    # a bright target with a dark shadow behind it (sonar-like pair)
    x0, y0 = rng.integers(20, size - 40, size=2)
    img[y0 : y0 + 18, x0 : x0 + 18] = 235
    img[y0 + 18 : y0 + 30, x0 : x0 + 18] = 20
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def seed() -> None:
    import uuid

    reset_inference_service()
    settings = get_settings()
    settings.ensure_dirs()
    repo = get_file_repository(settings)
    storage = get_storage_service(settings)

    # --- 1. stub model registered as active -------------------------------
    # (active ONLY if no other active model exists — never overrides a real trained model)
    models_dir = settings.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    pp_path = REPO_ROOT / "ml" / "configs" / "preprocessing" / "baseline_sonar.yaml"
    registry_path = models_dir / "registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.is_file() else {"models": []}
    has_other_active = any(
        m.get("status") == "active" and m.get("model_version") != "stub-demo-v1"
        for m in registry["models"]
    )
    stub_status = "shadow" if has_other_active else "active"
    if not any(m["model_version"] == "stub-demo-v1" for m in registry["models"]):
        registry["models"].append({
            "model_version": "stub-demo-v1",
            "architecture_family": "stub",
            "framework": "numpy",
            "checkpoint_path": "models/weights/stub-demo-v1/stub_checkpoint.json",
            "input_size": [640, 640],
            "class_map": {"0": "synthetic_target_a", "1": "synthetic_target_b"},
            "preprocess_config_ref": {
                "path": str(pp_path),
                "sha256": hashlib.sha256(pp_path.read_bytes()).hexdigest(),
            },
            "status": stub_status,
            "notes": "DEMO FIXTURE: StubDetector on synthetic patterns — not a trained sonar model.",
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        })
        registry_path.write_text(json.dumps(registry, indent=2))
        print(f"registered demo model: stub-demo-v1 (labelled as demo fixture; status={stub_status})")

    # --- 2. demo survey with synthetic tiles + synthetic nav track --------
    if any(d.get("name") == "DEMO synthetic survey" for d in repo._read("surveys").values()):
        print("demo survey already seeded; skipping")
        return

    n_tiles = 4
    buf = io.BytesIO()
    start = datetime(2026, 1, 15, 9, 0, 0, tzinfo=UTC)
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(n_tiles):
            zf.writestr(f"tile_{i:02d}.png", _demo_tile(seed=100 + i))
        # synthetic track along a straight line — DEMO ONLY, not a real survey
        lines = ["timestamp,latitude,longitude"]
        for i in range(n_tiles):
            t = start + timedelta(seconds=10 * i)
            lines.append(f"{t.isoformat()},{19.000 + 0.001 * i},{72.000 + 0.001 * i}")
        zf.writestr("nav.csv", "\n".join(lines) + "\n")

    zip_bytes = buf.getvalue()
    stored = storage.save_upload(zip_bytes, "demo_survey_SEEDED_FIXTURE.zip")

    from mlpipeline.io.navigation import parse_navigation

    survey_id = f"srv_{uuid.uuid4().hex[:16]}"
    image_ids: list[str] = []
    track_ref: str | None = None
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        with __import__("tempfile").TemporaryDirectory() as td:
            zf.extractall(td)
            root = Path(td)
            track = parse_navigation(root / "nav.csv")
            nav_status = "present"
            # Persist the parsed track as a DERIVED artifact (the original zip
            # in uploads/ stays immutable); the survey references it by relative
            # path so batch runs can geolocate against it.
            track_artifact = storage.save_artifact(
                (root / "nav.csv").read_bytes(), "navigation", f"{survey_id}_nav.csv"
            )
            track_ref = str(track_artifact.relative_to(settings.data_root.resolve()))
            for p in sorted(root.glob("tile_*.png")):
                raw = p.read_bytes()
                img_stored = storage.save_upload(raw, p.name)
                import cv2
                import numpy as np

                arr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                image_id = f"img_{img_stored['sha256'][:16]}_{img_stored['file_id'][4:12]}"
                repo.insert("sonar_images", {
                    "image_id": image_id,
                    "survey_id": survey_id,
                    "source_path": str(Path(img_stored["path"]).relative_to(settings.data_root.resolve())),
                    "sha256": img_stored["sha256"],
                    "format": "png",
                    "width": int(arr.shape[1]),
                    "height": int(arr.shape[0]),
                    "captured_at": None,
                    "acquisition_metadata": {
                        "original_filename": p.name,
                        "demo_fixture": True,
                    },
                    "geo_context_status": nav_status,
                    "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                })
                image_ids.append(image_id)

    repo.insert("surveys", {
        "survey_id": survey_id,
        "name": "DEMO synthetic survey",
        "source_archive": str(Path(stored["path"]).relative_to(settings.data_root.resolve())),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "image_count": len(image_ids),
        "images": image_ids,
        "navigation": {
            "status": nav_status,
            "format": track.format,
            "track_ref": track_ref,
            "sample_count": len(track.samples),
            "time_range": [track.samples[0].timestamp, track.samples[-1].timestamp] if track.samples else None,
            "warnings": ["SYNTHETIC DEMO NAVIGATION — not real survey data"] + track.warnings[:3],
        },
        "demo_fixture": True,
    })
    print(f"seeded demo survey: {survey_id} ({len(image_ids)} synthetic tiles, synthetic nav track)")
    print("run: make dev  →  open http://localhost:5173")


if __name__ == "__main__":
    seed()
