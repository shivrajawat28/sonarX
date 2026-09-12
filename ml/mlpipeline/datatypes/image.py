"""Image-side canonical datatypes: source records, processed images, scale bookkeeping.

Section 6.6 contract: what travels with a processed image (scale factors, applied
ops, config hash) is what makes post-processing able to map boxes back to source
coordinates exactly once.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from mlpipeline.datatypes.common import new_id, now_utc_iso


class SonarImage(BaseModel):
    """Metadata record for one stored sonar image (pixels live on disk, immutable)."""

    image_id: str = Field(default_factory=lambda: new_id("img"))
    survey_id: str | None = None
    source_path: str
    sha256: str
    format: str  # png | tiff | jpeg | ...
    width: int
    height: int
    captured_at: str | None = None
    acquisition_metadata: dict[str, Any] = Field(default_factory=dict)
    geo_context_status: Literal["present", "absent", "unparseable"] = "absent"
    created_at: str = Field(default_factory=now_utc_iso)


class ScaleFactors(BaseModel):
    """Letterbox bookkeeping: how processed space maps back to source space.

    source_x = (processed_x - pad_x) / scale ; likewise for y.
    """

    scale_x: float
    scale_y: float
    pad_x: float = 0.0
    pad_y: float = 0.0


class AppliedOp(BaseModel):
    """One preprocessing op as actually applied (params = the values used)."""

    op: str
    params: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0


class ImageRecord(BaseModel):
    """Input wrapper handed to the pipeline: pixels + provenance."""

    image: SonarImage
    array_sha256: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessedSonarImage(BaseModel):
    """Result of the preprocessing stage — the ONLY thing a Detector may consume.

    Note: pixels are intentionally not serialized here (kept in memory /
    stored separately as artifacts); this model carries the metadata trail.
    """

    image_id: str
    width: int  # processed-space width
    height: int  # processed-space height
    source_width: int
    source_height: int
    scale_factors: ScaleFactors
    applied_ops: list[AppliedOp] = Field(default_factory=list)
    config_hash: str
    config_name: str | None = None
    warnings: list[str] = Field(default_factory=list)

    def to_source_coords(self, x: float, y: float) -> tuple[float, float]:
        sf = self.scale_factors
        return (x - sf.pad_x) / sf.scale_x, (y - sf.pad_y) / sf.scale_y

    def to_processed_coords(self, x: float, y: float) -> tuple[float, float]:
        sf = self.scale_factors
        return x * sf.scale_x + sf.pad_x, y * sf.scale_y + sf.pad_y


# SonarImage re-export used by API layer via model_dump; ImageRecord is the input wrapper.
__all__ = [
    "SonarImage",
    "ImageRecord",
    "ProcessedSonarImage",
    "ScaleFactors",
    "AppliedOp",
]
