from datetime import date

import pytest

from snowtool.snowdb.snowline import snow_line_elevation
from snowtool.snowdb.zonal_stat_models import BandZoneRef, CompactStats, CompactZone


def band(lo: int, hi: int) -> BandZoneRef:
    return BandZoneRef(layer='terrain.elevation', min=lo, max=hi, unit='ft')


STATS = CompactStats(
    zone_layers=['terrain.elevation'],
    variables=['mean_snow_fraction_percent'],
    zones=[
        CompactZone(zone=[band(5000, 6000)], area_m2=4_102_331.0),
        CompactZone(zone=[band(6000, 7000)], area_m2=8_874_120.5),
        CompactZone(zone=[band(7000, 8000)], area_m2=6_218_904.2),
        CompactZone(zone=[band(8000, 9000)], area_m2=2_431_887.9),
    ],
    results={
        date(2026, 4, 15): [[71.2], [92.4], [98.1], [99.0]],
        date(2026, 5, 15): [[8.1], [41.6], [88.3], [96.7]],
        date(2026, 6, 15): [[0.0], [2.3], [19.4], [64.8]],
        date(2026, 8, 15): [[0.0], [1.7], [7.3], [20.2]],
    },
)


def test_crossing_interpolates_between_band_centres():
    lines = snow_line_elevation(STATS)
    may = lines[date(2026, 5, 15)]
    assert may is not None
    assert may.elevation_ft == pytest.approx(6679.9, abs=1.0)
    assert may.interpolation_span_ft == 1000


def test_last_pairing_breakpoint():
    june = snow_line_elevation(STATS)[date(2026, 6, 15)]
    assert june is not None
    assert june.elevation_ft == pytest.approx(8174.0, abs=1.0)


def test_no_crossing_when_all_bands_above_or_below_threshold():
    assert snow_line_elevation(STATS)[date(2026, 4, 15)] is None
    assert snow_line_elevation(STATS)[date(2026, 8, 15)] is None


def test_band_areas_not_swapped():
    may = snow_line_elevation(STATS)[date(2026, 5, 15)]
    assert may.lower_band_area_m2 == 8_874_120.5
    assert may.higher_band_area_m2 == 6_218_904.2


def test_custom_snowline_threshold():
    may = snow_line_elevation(STATS, threshold=65.0)[date(2026, 5, 15)]
    assert may is not None
    assert may.elevation_ft == pytest.approx(7001.0, abs=1.0)
