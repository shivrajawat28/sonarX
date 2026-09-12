"""Survey datatypes (Section 13)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from mlpipeline.datatypes.common import new_id, now_utc_iso

NavigationStatus = Literal["present", "absent", "unparseable"]


class NavigationInfo(BaseModel):
    status: NavigationStatus = "absent"
    format: str | None = None  # null when absent
    track_ref: str | None = None
    sample_count: int = 0
    time_range: list[str] | None = None  # [start, end] ISO strings when timestamps exist


class Survey(BaseModel):
    survey_id: str = Field(default_factory=lambda: new_id("srv"))
    name: str
    created_at: str = Field(default_factory=now_utc_iso)
    source_archive: str | None = None
    image_count: int = 0
    images: list[str] = Field(default_factory=list)  # SonarImage ids
    navigation: NavigationInfo = Field(default_factory=NavigationInfo)
    notes: str = ""


class SurveyManifest(BaseModel):
    """Serialization manifest for a survey stored under data/db (file repository)."""

    survey: Survey
    image_records: list[dict] = Field(default_factory=list)  # SonarImage dicts
