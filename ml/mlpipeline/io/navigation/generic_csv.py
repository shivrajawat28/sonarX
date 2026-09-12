"""Generic CSV navigation parser — the MVP sidecar contract (Section 10.2).

Accepted header (order-insensitive, extras preserved verbatim):
    timestamp | ping_index (at least one required)
    latitude, longitude (required, decimal degrees)
    heading_deg, altitude_m, speed_mps (optional)

``raw_row`` keeps the ORIGINAL row verbatim for traceability (Section 10.4).
"""
from __future__ import annotations

import csv
from pathlib import Path

from mlpipeline.datatypes.common import now_utc_iso
from mlpipeline.datatypes.navigation import NavigationSample, NavigationTrack
from mlpipeline.io.navigation.base import (
    NavigationParseError,
    register_navigation_provider,
)

REQUIRED = ("latitude", "longitude")
OPTIONAL_TIME = ("timestamp", "ping_index")
KNOWN_COLUMNS = set(REQUIRED) | set(OPTIONAL_TIME) | {"heading_deg", "altitude_m", "speed_mps"}


@register_navigation_provider
class GenericCSVParser:
    name = "generic_csv"

    def can_parse(self, source: Path) -> bool:
        if source.suffix.lower() != ".csv":
            return False
        try:
            with open(source, newline="", encoding="utf-8") as f:
                header = csv.DictReader(f).fieldnames
        except (OSError, csv.Error, UnicodeDecodeError):
            return False
        if not header:
            return False
        cols = {h.strip().lower() for h in header}
        return all(c in cols for c in REQUIRED) and any(c in cols for c in OPTIONAL_TIME)

    def parse(self, source: Path) -> NavigationTrack:
        samples: list[NavigationSample] = []
        warnings: list[str] = []
        with open(source, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for lineno, row in enumerate(reader, start=2):  # line 1 = header
                try:
                    clean = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
                    missing = [c for c in REQUIRED if not clean.get(c)]
                    if missing:
                        raise NavigationParseError(f"missing {missing}")
                    if not any(clean.get(c) for c in OPTIONAL_TIME):
                        raise NavigationParseError("need timestamp or ping_index")

                    lat = float(clean["latitude"])
                    lon = float(clean["longitude"])
                    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                        raise NavigationParseError(f"lat/lon out of range: {lat},{lon}")

                    samples.append(
                        NavigationSample(
                            index=len(samples),
                            timestamp=clean.get("timestamp") or None,
                            ping_index=int(clean["ping_index"]) if clean.get("ping_index") else None,
                            latitude=lat,
                            longitude=lon,
                            heading_deg=float(clean["heading_deg"]) if clean.get("heading_deg") else None,
                            altitude_m=float(clean["altitude_m"]) if clean.get("altitude_m") else None,
                            speed_mps=float(clean["speed_mps"]) if clean.get("speed_mps") else None,
                            raw_row=dict(row),  # verbatim evidence (Section 10.4)
                        )
                    )
                except (ValueError, NavigationParseError) as e:
                    # Skip malformed rows but RECORD them — never silently drop.
                    warnings.append(f"line {lineno}: {e}")

        if not samples:
            raise NavigationParseError(f"no valid navigation rows in {source.name}: {warnings[:5]}")
        return NavigationTrack(
            format="generic_csv", samples=samples, source_ref=str(source), warnings=warnings
        )
