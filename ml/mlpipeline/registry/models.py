"""Model registry: append-only index of model versions (models/registry.json).

Rules (Section 6.4):
- Entries are append-only; `register` refuses to overwrite an existing version.
- The file is Git-tracked; weights are not.
- `get` / `get_active` are read paths used by the detector adapters and backend.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from mlpipeline.datatypes.model import ModelVersion, ModelStatus


class RegistryError(Exception):
    pass


def repo_root() -> Path:
    """Repository root (this file lives at <root>/ml/mlpipeline/registry/models.py)."""
    return Path(__file__).resolve().parents[3]


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a registry path reference to a real file location.

    Registry entries may store either an absolute path (as written by an
    interactive registration on one machine) or a repository-relative one. An
    absolute path that no longer exists — the usual result of moving or syncing
    the repo — falls back to the same relative location under the repo root, so
    the registry stays portable instead of silently degrading to MODEL_UNAVAILABLE.
    """
    p = Path(path)
    if not p.is_absolute():
        return repo_root() / p
    if p.exists():
        return p
    # Absolute path from another checkout: try its suffixes as repo-relative
    # paths, longest first, and take the first that actually exists.
    parts = p.parts
    for i in range(1, len(parts)):
        candidate = repo_root().joinpath(*parts[i:])
        if candidate.exists():
            return candidate
    return p


class ModelRegistry:
    def __init__(self, registry_path: str | Path) -> None:
        self.path = Path(registry_path)
        self._lock = threading.Lock()
        self._cache: dict[str, ModelVersion] | None = None

    # -- reads -------------------------------------------------------------
    def _load(self) -> dict[str, ModelVersion]:
        if self._cache is not None:
            return self._cache
        if not self.path.is_file():
            self._cache = {}
            return self._cache
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            raise RegistryError(f"registry unreadable at {self.path}: {e}") from e
        items = raw.get("models", raw) if isinstance(raw, dict) else raw
        self._cache = {
            m["model_version"]: ModelVersion.model_validate(m) for m in items
        }
        return self._cache

    def invalidate(self) -> None:
        self._cache = None

    def list(self) -> list[ModelVersion]:
        return sorted(self._load().values(), key=lambda m: m.created_at)

    def get(self, model_version: str) -> ModelVersion | None:
        return self._load().get(model_version)

    def get_active(self) -> ModelVersion | None:
        for m in self.list():
            if m.status == "active":
                return m
        return None

    # -- writes (append-only) -----------------------------------------------
    def register(self, entry: ModelVersion, overwrite: bool = False) -> ModelVersion:
        """Append a model version. Refuses to silently overwrite history."""
        with self._lock:
            existing = self._load()
            if entry.model_version in existing and not overwrite:
                raise RegistryError(
                    f"model version '{entry.model_version}' already registered; "
                    f"new versions must use a new id (append-only registry)"
                )
            existing[entry.model_version] = entry
            payload = {"models": [m.model_dump() for m in existing.values()]}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.path)
            self._cache = None
            return entry

    def set_status(self, model_version: str, status: ModelStatus) -> ModelVersion:
        """Status transitions (e.g. shadow -> active) preserve the entry identity."""
        with self._lock:
            entries = self._load()
            if model_version not in entries:
                raise RegistryError(f"unknown model version '{model_version}'")
            updated = entries[model_version].model_copy(update={"status": status})
            entries[model_version] = updated
            payload = {"models": [m.model_dump() for m in entries.values()]}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.path)
            self._cache = None
            return updated


_default_registry: ModelRegistry | None = None


def get_registry_at(path: str | Path) -> ModelRegistry:
    return ModelRegistry(path)


def get_registry() -> ModelRegistry:
    """Default registry at <repo>/models/registry.json."""
    global _default_registry
    if _default_registry is None:
        root = Path(__file__).resolve().parents[3]
        _default_registry = ModelRegistry(root / "models" / "registry.json")
    return _default_registry
