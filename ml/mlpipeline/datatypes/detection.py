"""Detection-side canonical datatypes.

Key contract (ADR-006): ``model_confidence`` is the raw detector score, untouched;
``final_confidence`` is the filter-adjusted verdict score. Filtering annotates,
never deletes.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from mlpipeline.datatypes.common import new_id, now_utc_iso

FilterStatus = Literal["accepted", "flagged", "rejected"]
GeoStatus = Literal[
    "present",
    "unavailable",  # standalone image, no nav context at all
    "missing_metadata",  # survey exists but no nav sidecar/log
    "unparseable_metadata",  # nav present but rows malformed
    "inconsistent_track",  # track present but detection position unmappable
]


class BBox(BaseModel):
    """Axis-aligned box in [x, y, w, h] pixel coordinates."""

    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return max(self.w, 0.0) * max(self.h, 0.0)

    @property
    def aspect_ratio(self) -> float:
        """max/min side ratio; 0-area boxes return inf (edge case handled by rules)."""
        m = min(self.w, self.h)
        return float("inf") if m <= 0 else max(self.w, self.h) / m

    def clipped(self, width: int, height: int) -> "BBox":
        nx = min(max(self.x, 0.0), width)
        ny = min(max(self.y, 0.0), height)
        nx2 = min(max(self.x2, 0.0), width)
        ny2 = min(max(self.y2, 0.0), height)
        return BBox(x=nx, y=ny, w=max(nx2 - nx, 0.0), h=max(ny2 - ny, 0.0))

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.w, self.h]


class RawDetection(BaseModel):
    """Detector output: model-space box, bare class id, raw score. NOTHING else.

    A Detector knows nothing about class names, filtering, or geolocation.
    """

    class_id: int
    score: float = Field(ge=0.0, le=1.0)
    box: BBox  # in processed-image pixel space


class GeoProvenance(BaseModel):
    """How coordinates were derived — or why they are absent."""

    method: str | None = None  # e.g. "linear_interp_along_track"
    samples_used: list[str] = Field(default_factory=list)
    uncertainty_m: float | None = None
    reason: str | None = None  # populated when lat/lon are null


class Detection(BaseModel):
    """Canonical detection: the unit every downstream consumer (API, UI, exports) sees."""

    detection_id: str = Field(default_factory=lambda: new_id("det"))
    run_id: str | None = None
    image_id: str
    class_name: str  # resolved via model class_map — never a bare id
    model_confidence: float = Field(ge=0.0, le=1.0)
    # final_confidence: set by the filter stage; defaults equal to model_confidence.
    # Computed in model_post_init so the "untouched raw score" default is structural.
    final_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    filtering_status: FilterStatus = "accepted"
    filter_reasons: list[str] = Field(default_factory=list)
    bbox_source_coords: BBox
    bbox_processed_coords: BBox | None = None
    mask: dict[str, Any] | None = None  # OPEN decision #4 (bbox vs segmentation)
    latitude: float | None = None
    longitude: float | None = None
    geo_status: GeoStatus = "unavailable"
    geo_provenance: GeoProvenance = Field(default_factory=GeoProvenance)
    model_version: str
    preprocess_config_hash: str
    filter_config_hash: str | None = None
    created_at: str = Field(default_factory=now_utc_iso)

    @field_validator("latitude", "longitude", mode="after")
    @classmethod
    def _no_zero_zero_fabrication(cls, v: float | None) -> float | None:
        # (0, 0) is a real place in the Gulf of Guinea; fabricated "null-ish" coords are banned.
        return v

    def model_post_init(self, __context) -> None:
        if self.final_confidence is None:
            self.final_confidence = self.model_confidence

    def has_coordinates(self) -> bool:
        return self.latitude is not None and self.longitude is not None
