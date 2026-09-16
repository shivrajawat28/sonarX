"""Regression tests for the file-backed repository's concurrency behavior.

Encodes an observed defect: `GET /jobs/{id}` returned HTTP 500 with
`PermissionError: [Errno 13] ... data/db/jobs.json` while a background job
thread was updating that collection.

Root cause: reads took no lock, so a reader could open a collection file at the
exact moment another thread's `os.replace` swapped it. On Windows this raises
`PermissionError` because CPython opens files without FILE_SHARE_DELETE.
Transient external locks (OneDrive, antivirus) produce the same error.

These tests are deterministic: the contention is injected, not raced.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from backend.app.core.config import Settings
from backend.app.persistence.file_repository import FileRepository


@pytest.fixture()
def repo(tmp_path: Path) -> FileRepository:
    s = Settings()
    s.data_root = tmp_path / "data"
    s.models_dir = tmp_path / "models"
    s.ensure_dirs()
    return FileRepository(s)


class TestTransientPermissionError:
    def test_read_retries_a_transient_permission_error(self, repo: FileRepository, monkeypatch):
        """A single sharing violation must not surface as a 500."""
        repo.insert("jobs", {"job_id": "j1", "status": "running"})
        path = repo._path("jobs")

        real_read_text = Path.read_text
        calls = {"n": 0}

        def flaky_read_text(self, *a, **kw):
            if self == path and calls["n"] == 0:
                calls["n"] += 1
                raise PermissionError(13, "Permission denied", str(self))
            return real_read_text(self, *a, **kw)

        monkeypatch.setattr(Path, "read_text", flaky_read_text)
        assert repo.get("jobs", "j1")["status"] == "running"
        assert calls["n"] == 1  # retried exactly once

    def test_write_retries_a_transient_permission_error(self, repo: FileRepository, monkeypatch):
        """A locked destination during os.replace must be retried, not fatal."""
        repo.insert("jobs", {"job_id": "j1", "status": "pending"})
        real_replace = Path.replace
        calls = {"n": 0}

        def flaky_replace(self, target, *a, **kw):
            if Path(target) == repo._path("jobs") and calls["n"] == 0:
                calls["n"] += 1
                raise PermissionError(13, "Permission denied", str(target))
            return real_replace(self, target, *a, **kw)

        monkeypatch.setattr(Path, "replace", flaky_replace)
        repo.update("jobs", "j1", {"status": "succeeded"})
        assert repo.get("jobs", "j1")["status"] == "succeeded"
        assert calls["n"] == 1

    def test_persistent_permission_error_is_reported_not_corrupted(
        self, repo: FileRepository, monkeypatch
    ):
        """If the lock never clears we fail loudly — never silently drop data."""
        repo.insert("jobs", {"job_id": "j1", "status": "pending"})

        def always_denied(self, *a, **kw):
            if self == repo._path("jobs"):
                raise PermissionError(13, "Permission denied", str(self))
            raise AssertionError("unexpected path")

        monkeypatch.setattr(Path, "read_text", always_denied)
        with pytest.raises(RuntimeError, match="unreadable"):
            repo.get("jobs", "j1")

    def test_writes_stay_atomic_in_the_face_of_a_transient_lock(
        self, repo: FileRepository, monkeypatch
    ):
        """Retrying must not leave a half-written collection file behind."""
        repo.insert("jobs", {"job_id": "j1", "status": "pending"})
        real_replace = Path.replace
        calls = {"n": 0}

        def flaky_replace(self, target, *a, **kw):
            if Path(target) == repo._path("jobs") and calls["n"] < 2:
                calls["n"] += 1
                raise PermissionError(13, "Permission denied", str(target))
            return real_replace(self, target, *a, **kw)

        monkeypatch.setattr(Path, "replace", flaky_replace)
        repo.update("jobs", "j1", {"status": "succeeded"})
        # The on-disk file is complete, valid JSON — never a partial write.
        on_disk = json.loads(repo._path("jobs").read_text(encoding="utf-8"))
        assert on_disk["j1"]["status"] == "succeeded"
        assert list(on_disk) == ["j1"]


class TestConcurrentReadersAndWriters:
    def test_reads_never_observe_a_partial_file(self, repo: FileRepository):
        """Many readers against a writer must always see valid, complete JSON."""
        docs = {f"j{i}": {"job_id": f"j{i}", "status": "pending"} for i in range(200)}
        repo._write("jobs", docs)

        errors: list[BaseException] = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                try:
                    got = repo.list_all("jobs")
                    if got[1] != 200:
                        errors.append(AssertionError(f"torn read: {got[1]} docs"))
                except BaseException as e:  # noqa: BLE001
                    errors.append(e)
                    return

        def writer():
            for i in range(120):
                repo.update("jobs", "j0", {"status": f"s{i}"})

        threads = [threading.Thread(target=reader) for _ in range(4)]
        w = threading.Thread(target=writer)
        for t in threads:
            t.start()
        w.start()
        w.join()
        stop.set()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"concurrent access failed: {errors[:3]}"

    def test_get_does_not_race_a_concurrent_write(self, repo: FileRepository):
        """The exact failing path: GET /jobs/{id} while a job thread updates it."""
        repo.insert("jobs", {"job_id": "j1", "status": "pending"})
        errors: list[BaseException] = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                try:
                    assert repo.get("jobs", "j1") is not None
                except BaseException as e:  # noqa: BLE001
                    errors.append(e)
                    return

        def writer():
            for i in range(200):
                repo.update("jobs", "j1", {"progress_pct": i})

        readers = [threading.Thread(target=reader) for _ in range(4)]
        w = threading.Thread(target=writer)
        for t in readers:
            t.start()
        w.start()
        w.join()
        stop.set()
        for t in readers:
            t.join(timeout=5)

        assert not errors, f"GET-while-writing raised: {errors[:3]}"
