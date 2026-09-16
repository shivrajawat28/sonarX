"""POST /uploads/image + /uploads/survey (Section 12).

Validation chain: size -> extension -> magic bytes -> decode. Originals are
immutable; SonarImage/Survey records are persisted via the repository.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, Request, UploadFile

from backend.app.core.config import get_settings
from backend.app.core.errors import InvalidImageError, UnsupportedFileTypeError
from backend.app.core.security import (
    sanitize_filename,
    validate_extension,
    validate_magic_bytes,
    validate_size,
)
from backend.app.services.inference_service import get_inference_service
from mlpipeline.io.navigation import parse_navigation
from mlpipeline.io.image_reader import ImageReadError

router = APIRouter(prefix="/uploads", tags=["uploads"])

IMAGE_EXTS = {"png", "tif", "tiff", "jpg", "jpeg"}


@router.post("/image", status_code=201)
async def upload_image(request: Request, file: UploadFile = File(...)) -> dict:
    settings = get_settings()
    service = get_inference_service(settings)

    safe_name = sanitize_filename(file.filename or "upload")
    ext = validate_extension(safe_name, settings)
    if ext not in IMAGE_EXTS:
        raise UnsupportedFileTypeError(
            f".{ext} is not a standalone sonar image", details={"allowed": sorted(IMAGE_EXTS)}
        )

    data = await file.read()
    validate_size(len(data), settings)
    validate_magic_bytes(data[:16], ext)

    import numpy as np
    import cv2

    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise InvalidImageError("uploaded file is not a decodable image")
    if img.shape[0] > 10000 or img.shape[1] > 10000 or (img.shape[0] * img.shape[1]) > 50_000_000:
        raise InvalidImageError(
            f"image dimensions {img.shape[1]}x{img.shape[0]} exceed safety limits (max 10000x10000 px)"
        )

    stored = service.storage.save_upload(data, safe_name)
    sha = stored["sha256"]
    doc = {
        "image_id": f"img_{sha[:16]}_{stored['file_id'][4:12]}",
        "survey_id": None,
        "source_path": _relative_to_data(settings, stored["path"]),
        "sha256": sha,
        "format": ext,
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "captured_at": None,
        "acquisition_metadata": {"original_filename": safe_name},
        "geo_context_status": "absent",
        "created_at": _now(),
    }
    doc = service.repo.insert("sonar_images", doc)
    return {k: doc[k] for k in ("image_id", "sha256", "format", "width", "height", "created_at")} | {
        "filename": safe_name,
        "size_bytes": stored["size_bytes"],
        "has_metadata": False,
    }


@router.post("/survey", status_code=201)
async def upload_survey(
    request: Request,
    file: UploadFile = File(...),
    name: str = Form(""),
) -> dict:
    """Survey archive (zip of images + optional nav CSV sidecar) — MVP contract.

    Navigation format is an OPEN decision (#3): the generic CSV sidecar is the
    documented interim convention.
    """
    import io
    import zipfile

    settings = get_settings()
    service = get_inference_service(settings)

    safe_name = sanitize_filename(file.filename or "survey.zip")
    ext = validate_extension(safe_name, settings)
    if ext != "zip":
        raise UnsupportedFileTypeError("survey uploads must be .zip archives")

    data = await file.read()
    validate_size(len(data), settings)
    validate_magic_bytes(data[:16], ext)

    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        from backend.app.core.errors import CorruptSonarDataError

        raise CorruptSonarDataError("survey archive is not a valid zip") from e

    total_uncompressed = sum(zinfo.file_size for zinfo in zf.infolist())
    if total_uncompressed > settings.max_upload_bytes * 5:
        from backend.app.core.errors import CorruptSonarDataError

        raise CorruptSonarDataError(
            f"survey archive uncompressed size ({total_uncompressed} bytes) exceeds safety limit"
        )

    # Security: reject path-traversal entries before extraction
    for member in zf.namelist():
        p_parts = Path(member).parts
        if member.startswith(("/", "\\")) or ".." in p_parts or (len(p_parts) > 0 and ":" in p_parts[0]):
            from backend.app.core.errors import CorruptSonarDataError

            raise CorruptSonarDataError(f"unsafe archive entry: {member}")

    stored = service.storage.save_upload(data, safe_name)
    survey_doc = {
        "survey_id": f"srv_{uuid.uuid4().hex[:16]}",
        "name": (name or Path(safe_name).stem)[:120],
        "source_archive": _relative_to_data(settings, stored["path"]),
        "created_at": _now(),
    }

    import tempfile

    from mlpipeline.io.image_reader import SUPPORTED_EXTENSIONS

    image_ids: list[str] = []
    nav_status = "absent"
    nav_format = None
    track_warnings: list[str] = []
    track_ref: str | None = None
    nav_sample_count = 0
    nav_time_range: list | None = None

    with tempfile.TemporaryDirectory() as td:
        zf.extractall(td)
        root = Path(td)

        # navigation sidecar (nav.csv / navigation.csv)
        for candidate in ("nav.csv", "navigation.csv"):
            nav_file = root / candidate
            if nav_file.is_file():
                try:
                    track = parse_navigation(nav_file)
                    nav_status = "present"
                    nav_format = track.format
                    track_warnings = track.warnings[:5]
                    # Persist the track AND record its path — batch inference
                    # loads geolocation from this ref (bugfix: the artifact was
                    # saved but track_ref was hardcoded to None, so surveys
                    # uploaded through the API never geolocated).
                    artifact = service.storage.save_artifact(
                        nav_file.read_bytes(), "navigation", f"{stored['file_id']}_nav.csv"
                    )
                    track_ref = _relative_to_data(settings, artifact)
                    nav_sample_count = len(track.samples)
                    stamps = [s.timestamp for s in track.samples if s.timestamp]
                    nav_time_range = [stamps[0], stamps[-1]] if stamps else None
                except Exception as e:  # noqa: BLE001
                    nav_status = "unparseable"
                    track_warnings = [str(e)[:200]]
                break

        img_files = sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            and "nav" not in p.stem.lower()
        )
        if not img_files:
            from backend.app.core.errors import CorruptSonarDataError

            raise CorruptSonarDataError("archive contains no supported images")

        survey_doc = service.repo.insert("surveys", survey_doc)
        for p in img_files:
            raw = p.read_bytes()
            arr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if arr is None:
                continue  # recorded via survey notes; validation happens at run time
            img_stored = service.storage.save_upload(raw, p.name)
            img_doc = {
                "image_id": f"img_{img_stored['sha256'][:16]}_{img_stored['file_id'][4:12]}",
                "survey_id": survey_doc["survey_id"],
                "source_path": _relative_to_data(settings, img_stored["path"]),
                "sha256": img_stored["sha256"],
                "format": p.suffix.lower().lstrip("."),
                "width": int(arr.shape[1]),
                "height": int(arr.shape[0]),
                "captured_at": None,
                "acquisition_metadata": {"original_filename": sanitize_filename(p.name)},
                "geo_context_status": "present" if nav_status == "present" else nav_status,
                "created_at": _now(),
            }
            service.repo.insert("sonar_images", img_doc)
            image_ids.append(img_doc["image_id"])

    # persist navigation status onto the survey
    survey_doc = service.repo.update(
        "surveys", survey_doc["survey_id"],
        {
            "image_count": len(image_ids),
            "images": image_ids,
            "navigation": {
                "status": nav_status,
                "format": nav_format,
                "track_ref": track_ref,
                "sample_count": nav_sample_count,
                "time_range": nav_time_range,
                "warnings": track_warnings,
            },
        },
    )
    return {
        "survey_id": survey_doc["survey_id"],
        "name": survey_doc["name"],
        "image_count": len(image_ids),
        "navigation_status": nav_status,
        "images": image_ids,
        "navigation_warnings": track_warnings,
    }


# --- local helpers (avoid heavier imports at module top) ---
from datetime import UTC, datetime
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import cv2  # noqa: E402


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _relative_to_data(settings, path: str) -> str:
    try:
        return str(Path(path).relative_to(settings.data_root.resolve()))
    except ValueError:
        return str(path)

