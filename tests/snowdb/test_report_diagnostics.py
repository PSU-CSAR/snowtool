"""Unit tests for the read-only report builders in snowdb.diagnostics."""

from datetime import date

import numpy

from snowtool.snowdb import diagnostics
from snowtool.snowdb.aoi import AOI
from snowtool.snowdb.cog import write_cog
from snowtool.snowdb.constants import TILE_BBOX_TAG
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.db import SnowDb

from .conftest import TILE

# --- coverage / completeness -------------------------------------------------


def test_coverage_report_reports_gaps(dataset):
    for name in ('20180101', '20180103'):
        (dataset._cogs / name).mkdir()

    result = diagnostics.coverage_report(dataset)

    assert result.date_count == 2
    assert result.first_date == date(2018, 1, 1)
    assert result.last_date == date(2018, 1, 3)
    assert result.gaps == ((date(2018, 1, 2), date(2018, 1, 2)),)


def test_completeness_report_flags_incomplete_date(dataset, swe_cog):
    findings = diagnostics.completeness_report(dataset)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.date == date(2018, 4, 27)
    assert 'swe' not in finding.missing
    assert set(finding.missing) == set(dataset.spec.variables) - {'swe'}


def test_completeness_report_respects_date_window(dataset, swe_cog):
    # The only date (2018-04-27) is outside this window, so nothing is reported.
    assert diagnostics.completeness_report(dataset, start=date(2019, 1, 1)) == []


# --- missing-files -----------------------------------------------------------


def test_missing_artifacts_empty_for_created_dataset(dataset):
    assert diagnostics.missing_artifacts(dataset) == []


def test_missing_artifacts_reports_deleted_dem(dataset):
    dataset._dem.unlink()

    assert 'dem' in diagnostics.missing_artifacts(dataset)


# --- aoi-coverage ------------------------------------------------------------


def test_aoi_coverage_unrasterized_then_covered(
    tmp_path,
    spec,
    source_dem,
    aoi_geojson,
):
    import shutil

    SnowDb.initialize(tmp_path, [spec])
    ds = Dataset.create(spec, tmp_path / 'data' / 'test', source_dem)
    db = SnowDb(tmp_path, [spec])
    shutil.copy(aoi_geojson, db.aois_path / 'pp.geojson')

    before = diagnostics.aoi_coverage_report(db, ds)
    assert before.unrasterized == ('12345:MT:USGS',)
    assert before.orphan_rasters == ()

    ds.rasterize_aoi(AOI.from_geojson(aoi_geojson))
    after = diagnostics.aoi_coverage_report(db, ds)
    assert after.unrasterized == ()


def test_aoi_coverage_flags_orphan_raster(tmp_path, spec, source_dem, aoi_geojson):
    SnowDb.initialize(tmp_path, [spec])
    ds = Dataset.create(spec, tmp_path / 'data' / 'test', source_dem)
    db = SnowDb(tmp_path, [spec])  # no global AOIs
    ds.rasterize_aoi(AOI.from_geojson(aoi_geojson))

    result = diagnostics.aoi_coverage_report(db, ds)

    assert result.orphan_rasters == ('12345:MT:USGS',)


# --- aoi-health --------------------------------------------------------------


def test_aoi_health_all_healthy(dataset, aoi_geojson):
    dataset.rasterize_aoi(AOI.from_geojson(aoi_geojson))

    health = diagnostics.aoi_health_report(dataset)

    assert len(health) == 1
    assert health[0].ok is True
    assert health[0].issue is None


def test_aoi_health_reports_no_dem_overlap(dataset, grid):
    # An all-nodata AOI raster has no STATISTICS_* tags -> SNODASError on open.
    nodata = -9999.0
    write_cog(
        dataset._aoi_rasters / '99999_MT_USGS.tif',
        numpy.full((TILE, TILE), nodata, dtype=numpy.float32),
        transform=grid.base_grid[0, 0].transform,
        tile_size=TILE,
        nodata=nodata,
        tags={TILE_BBOX_TAG: '0 0 0 0'},
    )

    bad = [h for h in diagnostics.aoi_health_report(dataset) if not h.ok]
    assert len(bad) == 1
    assert 'DEM overlap' in bad[0].issue


def test_aoi_health_reports_missing_tile_bbox(dataset, grid):
    # A raster with valid data but no SNOWTOOL_TILE_BBOX tag -> ValueError on open.
    write_cog(
        dataset._aoi_rasters / '88888_MT_USGS.tif',
        numpy.full((TILE, TILE), 1000.0, dtype=numpy.float32),
        transform=grid.base_grid[0, 0].transform,
        tile_size=TILE,
        nodata=-9999.0,
    )

    bad = [h for h in diagnostics.aoi_health_report(dataset) if not h.ok]
    assert any('TILE_BBOX' in h.issue for h in bad)


# --- value-ranges / grid -----------------------------------------------------


def test_value_ranges_report(dataset, swe_cog):
    ranges = diagnostics.value_ranges_report(dataset, date(2018, 4, 27))

    assert len(ranges) == 1  # only swe present for this date
    swe = ranges[0]
    assert swe.variable == 'swe'
    assert swe.minimum == swe.maximum == swe.mean == 50  # uniform SWE_VALUE
    assert swe.nodata_pct == 0.0


def test_grid_report(dataset):
    result = diagnostics.grid_report(dataset)

    assert result.rows == 512
    assert result.cols == 512
    assert result.n_tiles == 4  # 2x2 tiles
    assert result.is_geographic is True
    assert result.cell_area_m2 is None
    left, _bottom, _right, top = result.extent
    assert left == -120.0
    assert top == 45.0
