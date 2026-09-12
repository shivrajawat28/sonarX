"""Step 9 tests: geolocation — CSV parser, interpolation, NULL-PROPAGATION.

Gate (Section 25, step 9): interpolation works; missing nav NEVER fabricates coords.
"""
from pathlib import Path

import pytest

from mlpipeline.datatypes.detection import BBox, Detection, GeoProvenance
from mlpipeline.datatypes.navigation import NavigationSample, NavigationTrack
from mlpipeline.geolocation.associator import Geolocator
from mlpipeline.geolocation.geometry import haversine_m, interpolate_position
from mlpipeline.io.navigation import NavigationParseError, parse_navigation


@pytest.fixture
def track() -> NavigationTrack:
    """Straight 2-sample track: (19.000, 72.000) -> (19.010, 72.000) (~1.11 km)."""
    return NavigationTrack(
        format="generic_csv",
        samples=[
            NavigationSample(index=0, ping_index=0, latitude=19.000, longitude=72.000, raw_row={"lat": "19.000"}),
            NavigationSample(index=1, ping_index=100, latitude=19.010, longitude=72.000, raw_row={"lat": "19.010"}),
        ],
    )


def _det(**kw) -> Detection:
    base = dict(
        image_id="img", class_name="alpha", model_confidence=0.8,
        bbox_source_coords=BBox(x=10, y=10, w=20, h=20),
        model_version="m", preprocess_config_hash="p",
    )
    base.update(kw)
    return Detection(**base)


class TestNeverFabricate:
    def test_no_track_all_null_with_reason(self):
        out = Geolocator(None).associate([_det()], fraction_of_image=0.5)
        assert out[0].latitude is None and out[0].longitude is None
        assert out[0].geo_status == "unavailable"
        assert out[0].geo_provenance.reason  # explains WHY

    def test_no_fraction_all_null_not_fabricated(self, track):
        """Track exists but reader gave no position -> NOT the track midpoint, NULL."""
        out = Geolocator(track).associate([_det()], fraction_of_image=None)
        assert out[0].latitude is None and out[0].longitude is None
        assert out[0].geo_status == "unparseable_metadata"

    def test_zero_length_track_inconsistent(self):
        t = NavigationTrack(format="generic_csv", samples=[
            NavigationSample(index=0, latitude=19.0, longitude=72.0),
            NavigationSample(index=1, latitude=19.0, longitude=72.0),
        ])
        out = Geolocator(t).associate([_det()], fraction_of_image=0.5)
        assert out[0].geo_status == "inconsistent_track"
        assert out[0].latitude is None

    def test_nulls_survive_serialization(self, track):
        out = Geolocator(None).associate([_det()], 0.5)[0]
        rt = Detection.model_validate_json(out.model_dump_json())
        assert rt.latitude is None and rt.longitude is None
        assert rt.geo_status == "unavailable"


class TestInterpolation:
    def test_fraction_zero_is_first_sample(self, track):
        out = Geolocator(track).associate([_det()], fraction_of_image=0.0)[0]
        assert out.latitude == pytest.approx(19.000, abs=1e-6)
        assert out.longitude == pytest.approx(72.000, abs=1e-6)
        assert out.geo_status == "present"
        assert out.geo_provenance.method == "linear_interp_along_track"
        assert len(out.geo_provenance.samples_used) == 2

    def test_fraction_one_is_last_sample(self, track):
        out = Geolocator(track).associate([_det()], fraction_of_image=1.0)[0]
        assert out.latitude == pytest.approx(19.010, abs=1e-6)

    def test_fraction_half_is_midpoint(self, track):
        out = Geolocator(track).associate([_det()], fraction_of_image=0.5)[0]
        assert out.latitude == pytest.approx(19.005, abs=1e-6)
        assert out.geo_provenance.uncertainty_m is not None

    def test_fraction_clamped_out_of_range(self, track):
        out = Geolocator(track).associate([_det()], fraction_of_image=7.0)[0]
        assert out.latitude == pytest.approx(19.010, abs=1e-6)  # clamped, not extrapolated

    def test_multi_sample_track_brackets_correctly(self):
        t = NavigationTrack(format="generic_csv", samples=[
            NavigationSample(index=i, latitude=19.0 + 0.01 * i, longitude=72.0)
            for i in range(5)
        ])
        out = Geolocator(t).associate([_det()], fraction_of_image=0.75)[0]
        assert out.latitude == pytest.approx(19.030, abs=1e-3)

    def test_haversine_sanity(self):
        # one degree of latitude ~ 111.19 km
        d = haversine_m(19.0, 72.0, 19.01, 72.0)
        assert d == pytest.approx(1112.0, rel=0.01)


class TestGenericCSVParser:
    def _csv(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "nav.csv"
        p.write_text(content, encoding="utf-8")
        return p

    def test_valid_csv_parses_with_raw_rows(self, tmp_path: Path):
        p = self._csv(tmp_path,
            "timestamp,latitude,longitude,heading_deg\n"
            "2026-09-01T10:00:00Z,19.000,72.000,90\n"
            "2026-09-01T10:00:01Z,19.010,72.000,91\n")
        track = parse_navigation(p)
        assert track.format == "generic_csv"
        assert len(track.samples) == 2
        assert track.samples[0].raw_row["heading_deg"] == "90"  # verbatim

    def test_ping_index_only_accepted(self, tmp_path: Path):
        p = self._csv(tmp_path, "ping_index,latitude,longitude\n0,19.0,72.0\n1,19.01,72.0\n")
        track = parse_navigation(p)
        assert len(track.samples) == 2
        assert track.samples[1].ping_index == 1

    def test_missing_required_column_rejected(self, tmp_path: Path):
        p = self._csv(tmp_path, "timestamp,latitude\n2026-01-01,19.0\n")
        with pytest.raises(NavigationParseError):
            parse_navigation(p)

    def test_out_of_range_coords_rejected(self, tmp_path: Path):
        p = self._csv(tmp_path, "timestamp,latitude,longitude\nt0,99.0,72.0\n")
        with pytest.raises(NavigationParseError):
            parse_navigation(p)

    def test_malformed_rows_skipped_but_recorded(self, tmp_path: Path):
        p = self._csv(tmp_path,
            "timestamp,latitude,longitude\n"
            "t0,19.0,72.0\n"
            "t1,not_a_number,72.0\n"
            "t2,19.02,72.0\n")
        track = parse_navigation(p)
        assert len(track.samples) == 2
        assert len(track.warnings) == 1 and "line 3" in track.warnings[0]

    def test_all_rows_malformed_is_parse_error(self, tmp_path: Path):
        p = self._csv(tmp_path, "timestamp,latitude,longitude\nt0,x,y\n")
        with pytest.raises(NavigationParseError, match="no valid"):
            parse_navigation(p)

    def test_non_csv_extension_rejected(self, tmp_path: Path):
        p = tmp_path / "nav.txt"
        p.write_text("timestamp,latitude,longitude\nt0,19.0,72.0\n")
        with pytest.raises(NavigationParseError):
            parse_navigation(p)
