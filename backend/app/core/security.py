"""Security guards (Section 20): file type/size validation, safe paths."""
from __future__ import annotations

import re
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.core.errors import FileTooLargeError, UnsupportedFileTypeError

# magic-byte signatures for allowed formats
MAGIC_BYTES: dict[str, list[bytes]] = {
    ".png": [b"\x89PNG\r\n\x1a\n"],
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".tif": [b"II*\x00", b"MM\x00*"],
    ".tiff": [b"II*\x00", b"MM\x00*"],
    ".zip": [b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"],  # survey archives
}

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: str) -> str:
    """Strip anything path-like or control-ish; keep a readable base name.

    The RESULT is metadata only — stored artifacts use server-generated ids.
    """
    base = Path(name or "upload").name
    base = base.replace("..", "_")
    cleaned = _UNSAFE.sub("_", base).strip("._")[:120]
    return cleaned or "upload"


def validate_extension(filename: str, settings: Settings) -> str:
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in settings.allowed_extensions_set:
        raise UnsupportedFileTypeError(
            f"extension '{ext}' not allowed",
            details={"allowed": sorted(settings.allowed_extensions_set)},
        )
    return ext


def validate_magic_bytes(head: bytes, ext: str) -> None:
    sigs = MAGIC_BYTES.get(f".{ext}")
    if not sigs:
        return  # no signature registered for this ext (e.g. generic .dat sidecars)
    if not any(head.startswith(sig) for sig in sigs):
        raise UnsupportedFileTypeError(
            f"file content does not match declared type .{ext} (magic-byte mismatch)"
        )


def validate_size(size_bytes: int, settings: Settings) -> None:
    if size_bytes > settings.max_upload_bytes:
        raise FileTooLargeError(
            f"upload {size_bytes} bytes exceeds limit {settings.max_upload_bytes}",
            details={"max_upload_mb": settings.max_upload_mb},
        )


def resolve_within(root: Path, *parts: str) -> Path:
    """Join and CONFINE to root — path-traversal prevention (Section 20).

    Uses real path ancestry (not a string prefix): a sibling directory whose name
    merely starts with the root's name (e.g. root `/data` vs `/data-evil`) must
    not pass.
    """
    candidate = (root.joinpath(*parts)).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and not candidate.is_relative_to(root_resolved):
        raise ValueError(f"path escapes DATA_ROOT: {candidate}")
    return candidate
