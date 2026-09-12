"""Navigation datatypes (Section 10.2).

``raw_row`` preserves the original metadata verbatim for traceability — we never
normalize away the source evidence.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NavigationSample(BaseModel):
    """One normalized position fix along a track."""

    index: int  # ordinal position in the track
    timestamp: str | None = None  # ISO-8601 when available
    ping_index: int | None = None  # sonar ping counter when available
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    heading_deg: float | None = Field(default=None, ge=0.0, le=360.0)
    altitude_m: float | None = None
    speed_mps: float | None = None
    raw_row: dict[str, Any] = Field(default_factory=dict)


class NavigationTrack(BaseModel):
    """Ordered set of navigation samples describing one survey track."""

    format: str  # e.g. "generic_csv" (OPEN decision #3 for vendor formats)
    samples: list[NavigationSample] = Field(default_factory=list)
    source_ref: str | None = None  # path/id of the original nav file
    warnings: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.samples) == 0

    def sample_ids(self) -> list[int]:
        return [s.index for s in self.samples]
