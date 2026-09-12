"""Storage service: artifact store operations (Section 15).

Uploads are IMMUTABLE originals; derived content goes to artifacts/.
All paths are server-generated and confined to DATA_ROOT (Section 20).
"""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.core.security import resolve_within


class StorageService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.ensure_dirs()

    # -- originals (immutable) ------------------------------------------------
    def save_upload(self, data: bytes, original_filename: str) -> dict:
        """Store an original upload; returns storage metadata. Never mutated after."""
        ext = Path(original_filename).suffix.lower().lstrip(".") or "bin"
        file_id = f"upl_{uuid.uuid4().hex[:20]}"
        day_dir = Path(__import__("datetime").datetime.now(__import__("datetime").UTC).strftime("%Y/%m"))
        target = resolve_within(self.settings.uploads_dir, *day_dir.parts, f"{file_id}.{ext}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return {
            "file_id": file_id,
            "path": str(target),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }

    # -- derived artifacts ----------------------------------------------------
    def save_artifact(self, data: bytes, *parts: str) -> Path:
        target = resolve_within(self.settings.artifacts_dir, *parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target

    def save_processed_image(self, arr, image_id: str, suffix: str = "png") -> Path:
        import cv2

        ok, buf = cv2.imencode(f".{suffix}", arr)
        if not ok:
            raise RuntimeError("failed to encode processed image")
        return self.save_artifact(buf.tobytes(), "preprocessed", f"{image_id}.{suffix}")

    def save_export(self, data: str | bytes, name: str) -> Path:
        payload = data.encode("utf-8") if isinstance(data, str) else data
        return self.save_artifact(payload, "exports", name) if False else self._write_export(payload, name)

    def _write_export(self, payload: bytes, name: str) -> Path:
        target = resolve_within(self.settings.exports_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target

    def open_artifact(self, rel_path: str) -> bytes:
        """Read a stored file by path (used for image/report downloads)."""
        p = Path(rel_path)
        if not p.is_absolute():
            p = resolve_within(self.settings.data_root, rel_path)
        else:
            resolve_within(self.settings.data_root, *p.relative_to(self.settings.data_root.resolve()).parts)
        if not p.is_file():
            from backend.app.core.errors import NotFoundError

            raise NotFoundError(f"artifact not found: {p.name}")
        return p.read_bytes()


def get_storage_service(settings: Settings) -> StorageService:
    return StorageService(settings)
