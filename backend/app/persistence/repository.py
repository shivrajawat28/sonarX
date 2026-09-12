"""Repository interfaces (Section 15): the storage seam.

File-backed implementation now (ADR-004); Mongo later behind MONGODB_ENABLED
with IDENTICAL document shapes — a repository swap, not a data-model change.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Repository(Protocol):
    """Document-store semantics over Section 13 collections."""

    def insert(self, collection: str, doc: dict[str, Any]) -> dict[str, Any]: ...

    def get(self, collection: str, doc_id: str) -> dict[str, Any] | None: ...

    def update(self, collection: str, doc_id: str, patch: dict[str, Any]) -> dict[str, Any] | None: ...

    def list(
        self,
        collection: str,
        page: int = 1,
        size: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Returns (items, total) with stable ordering (newest first)."""
        ...
