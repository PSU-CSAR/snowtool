"""Unit tests for the read-only report builders in snowdb.diagnostics."""

import json

from datetime import date

import numpy
import pytest

from snowtool.exceptions import AOICoverageError
from snowtool.snowdb import diagnostics
from snowtool.snowdb.aoi import AOI
from snowtool.snowdb.cog import write_cog
from snowtool.snowdb.constants import TILE_BBOX_TAG
from snowtool.snowdb.coverage import Coverage
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.manager import SnowDbManager

from ..conftest import make_snowdb
from .conftest import TILE


def _write_basin(records_dir, triplet, *, x0, y0, x1, y1):
    """Write an AOI record with a rectangular basin to ``records_dir``."""
    point = {'type': 'Point', 'coordinates': [(x0 + x1) / 2, (y0 + y1) / 2]}
    polygon = {
        'type': 'Polygon',
        'coordinates': [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }
    feature = {
        'type': 'GeometryCollection',
        'id': triplet,
        'geometries': [point, polygon],
        'properties': {'name': 'Basin', 'source': 'test'},
    }
    path = records_dir / f'{triplet.replace(":", "_")}.geojson'
    path.write_text(json.dumps(feature))
    return path

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


def test_missing_artifacts_reports_deleted_terrain(dataset):
    from snowtool.snowdb.terrain import ELEVATION

    dataset.zones['terrain'].layer_path(ELEVATION).unlink()

    assert 'terrain' in diagnostics.missing_artifacts(dataset)


def test_missing_artifacts_reports_deleted_landcover(dataset):
    from snowtool.snowdb.landcover import FOREST_COVER

    dataset.zones['landcover'].layer_path(FOREST_COVER).unlink()

    assert 'landcover' in diagnostics.missing_artifacts(dataset)


# --- aoi-coverage ------------------------------------------------------------


def test_aoi_coverage_unrasterized_then_covered(
    tmp_path,
    spec,
    aoi_geojson,
):
    import shutil

    SnowDbManager.initialize(tmp_path, [spec])
    ds = Dataset.create(spec, tmp_path / 'data' / 'test')
    db = make_snowdb(tmp_path, [spec])
    shutil.copy(aoi_geojson, db.aoi_records_path / 'pp.geojson')

    before = diagnostics.aoi_coverage_report(db, ds)
    assert before.unrasterized == ('12345:MT:USGS',)
    assert before.orphan_rasters == ()

    ds.rasterize_aoi(AOI.from_geojson(aoi_geojson))
    after = diagnostics.aoi_coverage_report(db, ds)
    assert after.unrasterized == ()


def test_aoi_coverage_flags_orphan_raster(tmp_path, spec, aoi_geojson):
    SnowDbManager.initialize(tmp_path, [spec])
    ds = Dataset.create(spec, tmp_path / 'data' / 'test')
    db = make_snowdb(tmp_path, [spec])  # no global AOIs
    ds.rasterize_aoi(AOI.from_geojson(aoi_geojson))

    result = diagnostics.aoi_coverage_report(db, ds)

    assert result.orphan_rasters == ('12345:MT:USGS',)


def test_aoi_coverage_classifies_full_partial_none(tmp_path, spec, aoi_geojson):
    # The synthetic grid spans lon [-120, -114.88], lat [39.88, 45].
    SnowDbManager.initialize(tmp_path, [spec])
    Dataset.create(spec, tmp_path / 'data' / 'test')
    db = make_snowdb(tmp_path, [spec])
    records = db.aoi_records_path
    # Fully inside.
    _write_basin(records, '100:MT:USGS', x0=-119.9, y0=44.9, x1=-119.0, y1=44.0)
    # Straddles the western edge -> partial.
    _write_basin(records, '200:MT:USGS', x0=-120.5, y0=44.9, x1=-119.5, y1=44.0)
    # Entirely east of the grid -> none.
    _write_basin(records, '300:MT:USGS', x0=-110.0, y0=44.9, x1=-109.0, y1=44.0)

    result = diagnostics.aoi_coverage_report(db, db.datasets['test'])

    assert result.partial == ('200:MT:USGS',)
    assert result.uncovered == ('300:MT:USGS',)


# --- query guard: SnowDb.require_aoi_coverage --------------------------------


@pytest.fixture
def guard_db(tmp_path, spec):
    """A SnowDb with three AOIs: full, partial, and uncovered by the grid."""
    SnowDbManager.initialize(tmp_path, [spec])
    Dataset.create(spec, tmp_path / 'data' / 'test')
    db = make_snowdb(tmp_path, [spec])
    records = db.aoi_records_path
    _write_basin(records, 'full:MT:USGS', x0=-119.9, y0=44.9, x1=-119.0, y1=44.0)
    _write_basin(records, 'part:MT:USGS', x0=-120.5, y0=44.9, x1=-119.5, y1=44.0)
    _write_basin(records, 'none:MT:USGS', x0=-110.0, y0=44.9, x1=-109.0, y1=44.0)
    return db


def test_guard_passes_full_coverage(guard_db):
    assert (
        guard_db.require_aoi_coverage('full:MT:USGS', 'test') is Coverage.FULL
    )


def test_guard_raises_on_partial(guard_db):
    with pytest.raises(AOICoverageError, match='partially covered'):
        guard_db.require_aoi_coverage('part:MT:USGS', 'test')


def test_guard_allow_partial_bypasses(guard_db):
    assert (
        guard_db.require_aoi_coverage(
            'part:MT:USGS', 'test', allow_partial=True,
        )
        is Coverage.PARTIAL
    )


def test_guard_raises_on_uncovered_despite_allow_partial(guard_db):
    with pytest.raises(AOICoverageError, match='not covered'):
        guard_db.require_aoi_coverage('none:MT:USGS', 'test', allow_partial=True)


# --- aoi-health --------------------------------------------------------------


def test_aoi_health_all_healthy(dataset, aoi_geojson):
    dataset.rasterize_aoi(AOI.from_geojson(aoi_geojson))

    health = diagnostics.aoi_health_report(dataset)

    assert len(health) == 1
    assert health[0].ok is True
    assert health[0].issue is None


def test_aoi_health_flags_empty_mask(dataset, grid):
    # AOI rasters are bare masks now (decoupled from the DEM). An all-zero mask
    # means the AOI polygon falls outside the grid -> flagged as an empty mask.
    write_cog(
        dataset._aoi_rasters / '99999_MT_USGS.tif',
        numpy.zeros((TILE, TILE), dtype=numpy.uint8),
        transform=grid.base_grid[0, 0].transform,
        tile_size=TILE,
        nodata=0,
        tags={TILE_BBOX_TAG: '0 0 0 0'},
        compute_stats=False,
    )

    bad = [h for h in diagnostics.aoi_health_report(dataset) if not h.ok]
    assert len(bad) == 1
    assert 'empty mask' in bad[0].issue


def test_aoi_health_reports_missing_tile_bbox(dataset, grid):
    # A raster with no SNOWTOOL_TILE_BBOX tag -> ValueError on open.
    write_cog(
        dataset._aoi_rasters / '88888_MT_USGS.tif',
        numpy.ones((TILE, TILE), dtype=numpy.uint8),
        transform=grid.base_grid[0, 0].transform,
        tile_size=TILE,
        nodata=0,
        compute_stats=False,
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
