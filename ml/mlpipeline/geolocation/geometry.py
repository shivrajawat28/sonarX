"""Track geometry: along-track distances and interpolation (Section 10.3)."""
from __future__ import annotations

import math

from mlpipeline.datatypes.navigation import NavigationSample

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def along_track_distances(track: list[NavigationSample]) -> list[float]:
    """Cumulative along-track distance (m) from the first sample."""
    out = [0.0]
    for prev, cur in zip(track, track[1:]):
        out.append(out[-1] + haversine_m(prev.latitude, prev.longitude, cur.latitude, cur.longitude))
    return out


def interpolate_position(
    a: NavigationSample, b: NavigationSample, fraction: float
) -> tuple[float, float]:
    """Linear interpolation between two samples; fraction in [0,1] from a to b."""
    fraction = min(max(fraction, 0.0), 1.0)
    lat = a.latitude + (b.latitude - a.latitude) * fraction
    lon = a.longitude + (b.longitude - a.longitude) * fraction
    return lat, lon


def heading_interpolated(a: NavigationSample, b: NavigationSample, fraction: float) -> float | None:
    """Interpolate heading with circular (shortest-arc) blending."""
    if a.heading_deg is None or b.heading_deg is None:
        return None
    h1, h2 = a.heading_deg, b.heading_deg
    delta = ((h2 - h1 + 180.0) % 360.0) - 180.0
    return (h1 + delta * fraction) % 360.0
