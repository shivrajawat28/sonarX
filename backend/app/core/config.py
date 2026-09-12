"""Settings (Section 11.1): env + .env, single source for all tunables."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # --- paths ---
    data_root: Path = Field(default_factory=lambda: Settings._anchor(Path("data")))
    models_dir: Path = Field(default_factory=lambda: Settings._anchor(Path("models")))
    active_model_version: str = ""  # empty = registry's single active entry

    @staticmethod
    def _anchor(v: Path) -> Path:
        """Anchor relative env paths (e.g. DATA_ROOT=./data) to the repo root.

        Without this, path-containment checks (resolve_within) compare an
        absolute artifact path against a CWD-relative root and wrongly fail.
        """
        p = Path(v)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        return p

    @field_validator("data_root", "models_dir", mode="after")
    @classmethod
    def _anchor_paths(cls, v: Path) -> Path:
        return cls._anchor(v)

    # --- inference ---
    device: str = "auto"  # auto | cpu | cuda
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    min_confidence_override: float = 0.05  # per-request override floor

    # --- uploads / security ---
    max_upload_mb: int = 100
    allowed_extensions: str = "png,tif,tiff,jpg,jpeg,zip"
    cors_origins: str = "http://localhost:5173"

    # --- persistence / jobs ---
    mongodb_enabled: bool = False
    mongodb_url: str = "mongodb://localhost:27017"
    async_survey_threshold: int = 25  # OPEN decision #7

    # --- app ---
    log_level: str = "INFO"
    app_version: str = "0.1.0"

    @property
    def allowed_extensions_set(self) -> set[str]:
        return {e.strip().lower().lstrip(".") for e in self.allowed_extensions.split(",") if e.strip()}

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def uploads_dir(self) -> Path:
        return self.data_root / "uploads"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_root / "artifacts"

    @property
    def reports_dir(self) -> Path:
        return self.data_root / "reports"

    @property
    def exports_dir(self) -> Path:
        return self.data_root / "exports"

    @property
    def db_dir(self) -> Path:
        return self.data_root / "db"

    @property
    def tmp_dir(self) -> Path:
        return self.data_root / "tmp"

    def ensure_dirs(self) -> None:
        for d in (self.uploads_dir, self.artifacts_dir, self.reports_dir,
                  self.exports_dir, self.db_dir, self.tmp_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
