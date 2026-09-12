#!/usr/bin/env python3
"""Build the survey archive fixture used by the browser E2E (geolocation path).

The archive contains REAL DRISHTI-SSS test tiles plus a navigation sidecar whose
coordinates are EXPLICITLY SYNTHETIC. It exists only to exercise the survey ->
batch -> geolocation plumbing in the UI. The track is never presented as a real
survey and the fixture is named `demo_survey_SYNTHETIC_NAV.zip` on purpose.

    python scripts/make_e2e_survey_fixture.py

Writes: e2e/fixtures/geo_survey_demo.zip
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "datasets/processed/drishti-sss/test/images"
OUT = ROOT / "e2e/fixtures/geo_survey_demo.zip"

# Synthetic straight-line track (0.001 deg per 10 s leg) — TEST DATA ONLY.
NAV = "\n".join(
    [
        "timestamp,latitude,longitude",
        "2026-01-15T09:00:00Z,18.9000,72.8000",
        "2026-01-15T09:00:10Z,18.9010,72.8010",
        "2026-01-15T09:00:20Z,18.9020,72.8020",
    ]
)


def main() -> int:
    tiles = sorted(SRC.glob("pipe_*.jpg"))[:3]
    if len(tiles) < 3:
        print(f"ERROR: need 3 pipe_*.jpg tiles under {SRC}")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in tiles:
            zf.writestr(p.name, p.read_bytes())
        zf.writestr("nav.csv", NAV + "\n")
        zf.writestr(
            "SYNTHETIC_NAV_NOTICE.txt",
            "The nav.csv track in this archive is SYNTHETIC test data.\n"
            "It exists to exercise the survey geolocation path; it is not a real survey.\n",
        )
    OUT.write_bytes(buf.getvalue())
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes, {len(tiles)} real tiles + synthetic nav.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
