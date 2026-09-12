"""Geolocator: associate detections with real navigation data — or nothing.

HARD RULE (Section 10.3): the pipeline never invents coordinates. Without a
usable track, lat/lon stay null and geo_status explains why.
"""
from __future__ import annotations

from mlpipeline.datatypes.detection import Detection, GeoProvenance, GeoStatus
from mlpipeline.datatypes.navigation import NavigationTrack
from mlpipeline.geolocation.geometry import (
    along_track_distances,
    heading_interpolated,
    interpolate_position,
)


class Geolocator:
    """Associates detections with a NavigationTrack by fractional along-track position.

    ``fraction`` is the detection's relative position along the image swath
    (0 = start of track coverage, 1 = end). The mapping from image row/ping to
    fraction depends on the reader (OPEN decision #3); callers pass what the
    reader extracted. A missing/invalid fraction yields explicit null geo.
    """

    def __init__(self, track: NavigationTrack | None = None) -> None:
        self.track = track

    def associate(
        self,
        detections: list[Detection],
        fraction_of_image: float | None = None,
    ) -> list[Detection]:
        if self.track is None or self.track.is_empty:
            return [
                d.model_copy(
                    update={
                        "geo_status": "unavailable",
                        "geo_provenance": GeoProvenance(
                            reason="no navigation data available for this source"
                        ),
                    }
                )
                for d in detections
            ]

        samples = self.track.samples
        dists = along_track_distances(samples)
        total = dists[-1]
        if total <= 0:
            return [
                d.model_copy(
                    update={
                        "geo_status": "inconsistent_track",
                        "geo_provenance": GeoProvenance(
                            reason="track has zero length (degenerate navigation data)"
                        ),
                    }
                )
                for d in detections
            ]

        if fraction_of_image is None:
            return [
                d.model_copy(
                    update={
                        "geo_status": "unparseable_metadata",
                        "geo_provenance": GeoProvenance(
                            reason="detection position along track unknown (reader did not provide)"
                        ),
                    }
                )
                for d in detections
            ]

        target = min(max(fraction_of_image, 0.0), 1.0) * total
        # find bracketing samples
        idx = next((i for i, d0 in enumerate(dists) if d0 >= target), len(dists) - 1)
        i0 = max(0, idx - 1)
        i1 = min(len(samples) - 1, max(i0 + 1, idx))
        seg = dists[i1] - dists[i0]
        fraction = 0.0 if seg <= 0 else (target - dists[i0]) / seg
        lat, lon = interpolate_position(samples[i0], samples[i1], fraction)
        heading = heading_interpolated(samples[i0], samples[i1], fraction)

        uncertainty = seg / 2.0 if seg > 0 else 0.0
        prov = GeoProvenance(
            method="linear_interp_along_track",
            samples_used=[f"nav_{samples[i0].index}", f"nav_{samples[i1].index}"],
            uncertainty_m=round(uncertainty, 2),
        )
        return [
            d.model_copy(
                update={
                    "latitude": lat,
                    "longitude": lon,
                    "geo_status": "present",
                    "geo_provenance": prov,
                }
            )
            for d in detections
        ]
