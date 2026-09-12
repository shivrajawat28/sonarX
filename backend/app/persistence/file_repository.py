"""File-backed repository: one JSON file per collection under DATA_ROOT/db (ADR-004).

Single-process assumption is explicit and documented (Section 19.4). Writes are
atomic (tmp + replace) so a crash cannot corrupt a collection file.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings

COLLECTIONS = ("surveys", "sonar_images", "detection_runs", "detections",
               "model_versions", "evaluation_runs", "reports", "jobs")


class FileRepository:
    def __init__(self, settings: Settings) -> None:
        self.db_dir = settings.db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, collection: str) -> Path:
        if collection not in COLLECTIONS:
            raise ValueError(f"unknown collection '{collection}'")
        return self.db_dir / f"{collection}.json"

    def _read(self, collection: str) -> dict[str, dict[str, Any]]:
        p = self._path(collection)
        if not p.is_file():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            raise RuntimeError(f"collection '{collection}' unreadable: {e}") from e

    def _write(self, collection: str, docs: dict[str, dict[str, Any]]) -> None:
        p = self._path(collection)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(docs, indent=2), encoding="utf-8")
        tmp.replace(p)

    # -- Repository protocol -------------------------------------------------
    def insert(self, collection: str, doc: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            docs = self._read(collection)
            doc_id = doc.get("id") or _doc_id(doc, collection)
            if doc_id in docs:
                raise RuntimeError(f"{collection}/{doc_id} already exists")
            doc = {**doc, "id": doc_id}
            docs[doc_id] = doc
            self._write(collection, docs)
            return doc

    def get(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        return self._read(collection).get(doc_id)

    def update(self, collection: str, doc_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            docs = self._read(collection)
            if doc_id not in docs:
                return None
            docs[doc_id].update(patch)
            self._write(collection, docs)
            return docs[doc_id]

    def list(
        self,
        collection: str,
        page: int = 1,
        size: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        docs = self._read(collection)
        items = list(docs.values())
        for key, value in (filters or {}).items():
            if value is None:
                continue
            items = [d for d in items if d.get(key) == value]
        # newest first by created_at then id — stable pagination
        items.sort(key=lambda d: (d.get("created_at") or "", d.get("id") or ""), reverse=True)
        total = len(items)
        # The 200-document cap applies to the page LENGTH as well as the offset:
        # slicing with the caller's raw `size` silently returned unbounded pages
        # for any request above the cap.
        effective = max(min(size, 200), 1)
        start = (max(page, 1) - 1) * effective
        return items[start : start + effective], total

    def list_all(
        self,
        collection: str,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Every matching document, unpaginated.

        Used by exports and reports, where a page cap would silently truncate
        the artifact. Request-serving endpoints keep using `list` so they stay
        bounded.
        """
        docs = self._read(collection)
        items = list(docs.values())
        for key, value in (filters or {}).items():
            if value is None:
                continue
            items = [d for d in items if d.get(key) == value]
        items.sort(key=lambda d: (d.get("created_at") or "", d.get("id") or ""), reverse=True)
        return items, len(items)


_COLLECTION_ID_KEYS: dict[str, tuple[str, ...]] = {
    "surveys": ("survey_id",),
    "sonar_images": ("image_id",),
    "detection_runs": ("run_id",),
    "detections": ("detection_id",),
    "model_versions": ("model_version",),
    "evaluation_runs": ("eval_run_id",),
    "reports": ("report_id",),
    "jobs": ("job_id",),
}


def _doc_id(doc: dict[str, Any], collection: str) -> str:
    """Map Section-13 document id fields onto the storage `id` key.

    Per-collection key lists (a detection carries image_id AND detection_id —
    only its own detection_id identifies the document).
    """
    for key in _COLLECTION_ID_KEYS[collection]:
        if doc.get(key):
            return str(doc[key])
    raise ValueError(f"document in '{collection}' has no id field")


_repo: FileRepository | None = None


def get_file_repository(settings: Settings) -> FileRepository:
    global _repo
    if _repo is None:
        _repo = FileRepository(settings)
    return _repo


def reset_file_repository() -> None:
    """Test helper: force re-creation with fresh settings."""
    global _repo
    _repo = None
