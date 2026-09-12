"""Shared API schemas (Section 12 contract rules: envelope, pagination, request_id)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str = ""


class ErrorEnvelope(BaseModel):
    error: ErrorDetail


class Page(BaseModel):
    page: int = Field(default=1, ge=1)
    size: int = Field(default=20, ge=1, le=200)
