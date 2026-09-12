"""IDs and time helpers shared by all canonical datatypes.

Server-generated IDs (never user-controlled — part of the path-traversal
defense, see backend/app/core/security.py) and UTC timestamps.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime


def now_utc_iso() -> str:
    """Current UTC time as ISO-8601 string with second precision."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    """Prefixed, sortable-enough unique id, e.g. img_01H9... (server-generated only)."""
    return f"{prefix}_{uuid.uuid4().hex[:20]}"
