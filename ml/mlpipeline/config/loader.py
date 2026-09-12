"""Config loading: YAML -> validated Pydantic models, with content hashing.

Reproducibility contract (ADR-003): the sha256 of the canonical file content is
the config's identity; it is stored with model versions and in every result.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class ConfigError(Exception):
    """Raised when a config file is missing, unreadable, or fails validation.

    Message includes the validation details so a student can fix the YAML fast.
    """


def config_sha256(data: bytes) -> str:
    """Content hash of raw config bytes (canonical identity)."""
    return hashlib.sha256(data).hexdigest()


def sha256_of_file(path: Path) -> str:
    return config_sha256(Path(path).read_bytes())


def load_config_file(path: str | Path, model_cls: type[T]) -> tuple[T, str]:
    """Load a YAML config file into a validated model.

    Returns:
        (validated model, content sha256)

    Raises:
        ConfigError: file missing / invalid YAML / schema validation failure
            (with per-field error details).
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        raw = p.read_bytes()
    except OSError as e:
        raise ConfigError(f"config file unreadable: {p} ({e})") from e

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {p}: {e}") from e

    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping, got {type(data).__name__} in {p}")

    try:
        model = model_cls.model_validate(data)
    except ValidationError as e:
        details = "; ".join(
            f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        raise ConfigError(f"config validation failed for {p}: {details}") from e

    return model, config_sha256(raw)


def save_config_yaml(model: BaseModel, path: str | Path) -> str:
    """Persist a config model as YAML; returns the content hash of what was written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(model.model_dump(), sort_keys=False)
    p.write_bytes(payload.encode("utf-8"))  # write_bytes avoids OS line-ending conversion
    return config_sha256(p.read_bytes())
