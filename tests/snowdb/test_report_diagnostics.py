"""Unit tests for the read-only report builders in snowdb.diagnostics."""

import json
import shutil

from datetime import date

import numpy
import pytest
import rasterio

from snowtool.exceptions import PourpointCoverageError
from snowtool.snowdb import diagnostics
from snowtool.snowdb.constants import TILE_BBOX_TAG
from snowtool.snowdb.coverage import Coverage
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.manager import SnowDbManager
from snowtool.snowdb.pourpoint import Pourpoint
from snowtool.snowdb.raster.cog import write_cog
from snowtool.snowdb.spec import DatasetSpec
from snowtool.snowdb.zones.landcover import FOREST_COVER
from snowtool.snowdb.zones.terrain import ELEVATION

from ..conftest import make_snowdb, snodas_swe_name, write_landcover, write_terrain
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


def test_missing_dates_defaults_to_first_ingested_through_today(dataset):
    for name in ('20180101', '20180103'):
        (dataset._cogs / name).mkdir()

    result = diagnostics.missing_dates(dataset, end=date(2018, 1, 4))

    assert result == [date(2018, 1, 2), date(2018, 1, 4)]


def test_missing_dates_requires_start_when_no_ingested_dates(dataset):
    with pytest.raises(ValueError, match='no ingested dates'):
        diagnostics.missing_dates(dataset, end=date(2018, 1, 1))


def test_missing_dates_start_after_end_is_empty(dataset):
    for name in ('20180101',):
        (dataset._cogs / name).mkdir()

    assert (
        diagnostics.missing_dates(
            dataset,
            start=date(2018, 1, 5),
            end=date(2018, 1, 1),
        )
        == []
    )


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

    dataset.zones['terrain'].layer_path(ELEVATION).unlink()

    # The finding names the provider and the specific absent layer file.
    finding = next(
        m for m in diagnostics.missing_artifacts(dataset) if m.startswith('terrain')
    )
    assert ELEVATION.filename in finding


def test_missing_artifacts_reports_deleted_landcover(dataset):

    dataset.zones['landcover'].layer_path(FOREST_COVER).unlink()

    finding = next(
        m for m in diagnostics.missing_artifacts(dataset) if m.startswith('landcover')
    )
    assert FOREST_COVER.filename in finding


def test_stale_format_zone_layers_empty_for_current_dataset(dataset):
    # Freshly built sets carry the current format version -> no findings.
    assert diagnostics.stale_format_zone_layers(dataset) == []


def test_stale_format_zone_layers_flags_an_old_format(dataset):
    # Simulate a format bump: the provider now expects a newer version than what
    # is stamped on the built terrain set, so it is flagged for a rebuild.
    terrain = dataset.zones['terrain']
    stamped = terrain.stored_format_version()
    terrain.format_version = stamped + 1

    findings = diagnostics.stale_format_zone_layers(dataset)

    assert [(f.provider, f.stored, f.expected) for f in findings] == [
        ('terrain', stamped, stamped + 1),
    ]


def test_stale_format_zone_layers_skips_unbuilt_sets(dataset):
    # An unbuilt set is a missing-artifact finding, not a stale-format one.

    dataset.zones['landcover'].layer_path(FOREST_COVER).unlink()

    providers = {f.provider for f in diagnostics.stale_format_zone_layers(dataset)}
    assert 'landcover' not in providers


# --- pourpoint-coverage ------------------------------------------------------------


def test_pourpoint_coverage_unrasterized_then_covered(
    tmp_path,
    spec,
    pourpoint_geojson,
):

    SnowDbManager.initialize(tmp_path, [spec])
    ds = Dataset.create(spec, tmp_path / 'data' / 'test')
    db = make_snowdb(tmp_path, [spec])
    shutil.copy(pourpoint_geojson, db.pourpoint_records_path / 'pp.geojson')

    before = diagnostics.pourpoint_coverage_report(db, ds)
    assert before.unrasterized == ('12345:MT:USGS',)
    assert before.orphan_rasters == ()

    ds.rasterize_aoi(Pourpoint.from_geojson(pourpoint_geojson))
    after = diagnostics.pourpoint_coverage_report(db, ds)
    assert after.unrasterized == ()


def test_pourpoint_coverage_flags_orphan_raster(tmp_path, spec, pourpoint_geojson):
    SnowDbManager.initialize(tmp_path, [spec])
    ds = Dataset.create(spec, tmp_path / 'data' / 'test')
    db = make_snowdb(tmp_path, [spec])  # no global AOIs
    ds.rasterize_aoi(Pourpoint.from_geojson(pourpoint_geojson))

    result = diagnostics.pourpoint_coverage_report(db, ds)

    assert result.orphan_rasters == ('12345:MT:USGS',)


def test_pourpoint_coverage_classifies_full_partial_none(
    tmp_path,
    spec,
    pourpoint_geojson,
):
    # The synthetic grid spans lon [-120, -114.88], lat [39.88, 45].
    SnowDbManager.initialize(tmp_path, [spec])
    Dataset.create(spec, tmp_path / 'data' / 'test')
    db = make_snowdb(tmp_path, [spec])
    records = db.pourpoint_records_path
    # Fully inside.
    _write_basin(records, '100:MT:USGS', x0=-119.9, y0=44.9, x1=-119.0, y1=44.0)
    # Straddles the western edge -> partial.
    _write_basin(records, '200:MT:USGS', x0=-120.5, y0=44.9, x1=-119.5, y1=44.0)
    # Entirely east of the grid -> none.
    _write_basin(records, '300:MT:USGS', x0=-110.0, y0=44.9, x1=-109.0, y1=44.0)

    result = diagnostics.pourpoint_coverage_report(db, db.datasets['test'])

    assert result.partial == ('200:MT:USGS',)
    assert result.uncovered == ('300:MT:USGS',)


# --- query guard: SnowDb.require_pourpoint_coverage --------------------------------


@pytest.fixture
def guard_db(tmp_path, spec):
    """A SnowDb with three AOIs: full, partial, and uncovered by the grid."""
    SnowDbManager.initialize(tmp_path, [spec])
    Dataset.create(spec, tmp_path / 'data' / 'test')
    db = make_snowdb(tmp_path, [spec])
    records = db.pourpoint_records_path
    _write_basin(records, 'full:MT:USGS', x0=-119.9, y0=44.9, x1=-119.0, y1=44.0)
    _write_basin(records, 'part:MT:USGS', x0=-120.5, y0=44.9, x1=-119.5, y1=44.0)
    _write_basin(records, 'none:MT:USGS', x0=-110.0, y0=44.9, x1=-109.0, y1=44.0)
    # The coverage guard loads pourpoints through the index (availability gate),
    # so the records have to be indexed to be queryable.
    SnowDbManager(db).reindex_pourpoints()
    return db


def test_guard_passes_full_coverage(guard_db):
    assert guard_db.require_pourpoint_coverage('full:MT:USGS', 'test') is Coverage.FULL


def test_guard_raises_on_partial(guard_db):
    with pytest.raises(PourpointCoverageError, match='partially covered'):
        guard_db.require_pourpoint_coverage('part:MT:USGS', 'test')


def test_guard_allow_partial_bypasses(guard_db):
    assert (
        guard_db.require_pourpoint_coverage(
            'part:MT:USGS',
            'test',
            allow_partial=True,
        )
        is Coverage.PARTIAL
    )


def test_guard_raises_on_uncovered_despite_allow_partial(guard_db):
    with pytest.raises(PourpointCoverageError, match='not covered'):
        guard_db.require_pourpoint_coverage('none:MT:USGS', 'test', allow_partial=True)


# --- aoi-health --------------------------------------------------------------


def test_aoi_health_all_healthy(dataset, pourpoint_geojson):
    dataset.rasterize_aoi(Pourpoint.from_geojson(pourpoint_geojson))

    health = diagnostics.aoi_health_report(dataset)

    assert health == []


def test_aoi_health_flags_empty_aoi(dataset, grid):
    # AOI rasters carry per-pixel cell area now (decoupled from the DEM). An
    # all-zero raster means the AOI polygon falls outside the grid -> flagged.
    write_cog(
        dataset._aoi_rasters / '99999_MT_USGS.tif',
        numpy.zeros((TILE, TILE), dtype=numpy.float32),
        transform=grid.base_grid[0, 0].transform,
        tile_size=TILE,
        nodata=0,
        tags={TILE_BBOX_TAG: '0 0 0 0'},
        compute_stats=False,
    )

    bad = diagnostics.aoi_health_report(dataset)
    assert len(bad) == 1
    assert 'empty AOI' in bad[0].issue


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

    bad = diagnostics.aoi_health_report(dataset)
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


def test_dataset_info_report(tmp_path, spec):
    # A SnowDb-bound dataset (not the bare `dataset` fixture): `active` reads off
    # `snowdb.datasets`, so the report needs the two objects wired together the
    # way `dataset info` resolves them.
    db = make_snowdb(tmp_path, [spec])
    ds = db['test']
    write_terrain(ds)
    write_landcover(ds)

    result = diagnostics.dataset_info_report(db, ds)

    assert result.name == 'test'
    assert result.active is True
    assert result.present is True
    assert result.is_geographic is True
    assert result.cell_area_m2 is None  # geographic grid -> per-pixel area raster
    assert result.rows == result.cols == 512
    assert result.n_tiles == 4
    assert result.min_elevation_m == -100.0
    assert result.max_elevation_m == 4500.0
    assert result.variables == (
        'average_temp',
        'depth',
        'precip_liquid',
        'precip_solid',
        'runoff',
        'sublimation',
        'sublimation_blowing',
        'swe',
    )
    assert result.zone_layers['terrain']['present'] is True
    assert result.zone_layers['landcover']['present'] is True
    assert result.zone_layers['terrain']['hash'] is not None
    assert result.zones['terrain']['aspect'] is None
    assert result.zones['terrain']['elevation'] == {'band_step_ft': 1000}
    assert result.cogs is False  # no dates ingested
    assert result.aoi_rasters is False
    assert result.date_count == 0
    assert result.first_date is None
    assert result.last_date is None


# --- grid validation ---------------------------------------------------------


def test_grid_validation_clean_when_cog_matches(dataset, swe_cog):
    # swe_cog is written on the declared grid (matching transform + 512x512).
    assert diagnostics.grid_validation_report(dataset) == []


def test_grid_validation_skipped_without_a_cog(dataset):
    # No variable COG ingested yet -> grid check has nothing to compare against.
    assert diagnostics.grid_validation_report(dataset) == []


def test_grid_validation_flags_shape_mismatch(dataset):

    date_dir = dataset._cogs / '20180101'
    date_dir.mkdir(parents=True)
    # A 256x256 COG on a 512x512 declared grid (transform still matches origin/px).
    write_cog(
        date_dir / f'{snodas_swe_name("20180101")}.tif',
        numpy.zeros((256, 256), dtype=numpy.int16),
        transform=dataset.grid.base_grid.transform,
        tile_size=TILE,
    )

    issues = diagnostics.grid_validation_report(dataset)

    assert any('512x512' in issue and 'is 256x256' in issue for issue in issues)


def test_grid_validation_flags_transform_mismatch(dataset):

    date_dir = dataset._cogs / '20180101'
    date_dir.mkdir(parents=True)
    # Right shape, but the origin is shifted a full degree off the declared grid.
    shifted = dataset.grid.base_grid.transform * rasterio.Affine.translation(0, 0)
    shifted = rasterio.Affine(
        shifted.a,
        shifted.b,
        shifted.c + 1.0,
        shifted.d,
        shifted.e,
        shifted.f,
    )
    write_cog(
        date_dir / f'{snodas_swe_name("20180101")}.tif',
        numpy.zeros((512, 512), dtype=numpy.int16),
        transform=shifted,
        tile_size=TILE,
    )

    issues = diagnostics.grid_validation_report(dataset)

    assert any('transform' in issue for issue in issues)


def test_grid_validation_flags_ingester_without_variables(tmp_path, spec):

    class _Ingester:
        def ingest(self, source, dataset, *, force=False, **_):  # pragma: no cover
            from snowtool.snowdb.ingest import IngestResult

            return IngestResult(ingested=[], skipped=[])

    bare = DatasetSpec(
        name='bare',
        grid_params=spec.grid_params,
        variables=(),
        ingester=_Ingester(),
    )
    ds = Dataset(bare, tmp_path / 'bare', ())

    assert diagnostics.grid_validation_report(ds) == [
        'has an ingester but declares no variables',
    ]
