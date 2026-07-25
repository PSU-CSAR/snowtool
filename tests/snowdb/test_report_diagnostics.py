"""Unit tests for the read-only report builders in snowdb.diagnostics."""

import contextlib
import shutil

from datetime import date

import numpy
import pytest
import rasterio

from snowtool.exceptions import PourpointCoverageError
from snowtool.snowdb import diagnostics
from snowtool.snowdb import issues as issues_mod
from snowtool.snowdb.aoi_raster import aoi_provenance, aoi_raster_issues
from snowtool.snowdb.constants import AOI_HASH_TAG, AOI_MASK_NODATA, TILE_BBOX_TAG
from snowtool.snowdb.coverage import Coverage
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.datasets import config_from_spec
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.manager import SnowDbManager
from snowtool.snowdb.pourpoint import Pourpoint
from snowtool.snowdb.raster.cog import write_cog
from snowtool.snowdb.spec import DatasetSpec
from snowtool.snowdb.zones.landcover_layers import FOREST_COVER
from snowtool.snowdb.zones.terrain_layers import ELEVATION

from ..conftest import (
    ORIGIN_X,
    PX,
    TILE,
    make_snowdb,
    register_dataset_config,
    snodas_swe_name,
    synthetic_grid,
    write_landcover,
    write_pourpoint_record,
    write_swe_cog,
    write_terrain,
)


def _write_basin(records_dir, triplet, *, x0, y0, x1, y1):
    """Write an AOI record with a rectangular basin to ``records_dir``."""
    return write_pourpoint_record(
        records_dir / f'{triplet.replace(":", "_")}.geojson',
        triplet,
        box=(x0, y0, x1, y1),
        point=((x0 + x1) / 2, (y0 + y1) / 2),
        properties={'name': 'Basin', 'source': 'test'},
    )


@pytest.fixture
def created_db(tmp_path, spec):
    """An initialized root with the synthetic dataset created, registered on disk,
    and bound (read side).

    The shared "initialize + register + bind a SnowDb" trio for the coverage
    reports: returns ``(db, ds)`` where ``ds`` is the created dataset and ``db``
    the reader over the same root. Registered on disk (not just inline) so a
    ``PourpointManager`` write against ``db`` -- which now derives its coverage
    domains from a fresh on-disk open -- still sees ``'test'``.
    """
    manager = SnowDbManager.initialize(tmp_path)
    register_dataset_config(manager, 'test', config_from_spec(spec))
    db = SnowDb.open(tmp_path)
    return db, db['test']


# --- coverage / completeness -------------------------------------------------


@pytest.mark.parametrize(
    ('ingested', 'kwargs', 'expected'),
    [
        # No explicit start -> defaults to the first ingested date, through end;
        # the interior gap and the (absent) end date are both reported.
        pytest.param(
            ('20180101', '20180103'),
            {'end': date(2018, 1, 4)},
            [date(2018, 1, 2), date(2018, 1, 4)],
            id='defaults-start-to-first-ingested',
        ),
        # A fully contiguous span has no gap -> nothing missing.
        pytest.param(
            ('20180101', '20180102', '20180103'),
            {'start': date(2018, 1, 1), 'end': date(2018, 1, 3)},
            [],
            id='contiguous-is-empty',
        ),
        # An inverted window (start after end) yields no dates at all.
        pytest.param(
            ('20180101',),
            {'start': date(2018, 1, 5), 'end': date(2018, 1, 1)},
            [],
            id='start-after-end-is-empty',
        ),
    ],
)
def test_missing_dates(dataset, ingested, kwargs, expected):
    for name in ingested:
        (dataset._cogs / name).mkdir()

    assert diagnostics.missing_dates(dataset, **kwargs) == expected


def test_missing_dates_requires_start_when_no_ingested_dates(dataset):
    with pytest.raises(ValueError, match='no ingested dates'):
        diagnostics.missing_dates(dataset, end=date(2018, 1, 1))


def test_completeness_report_flags_incomplete_date(dataset, swe_cog):
    findings = diagnostics.completeness_report(dataset)

    assert len(findings) == 1
    finding = findings[0]
    assert finding['check'] == 'dates'
    assert finding['target'] == '2018-04-27'
    # The issue names every missing variable except the one present (swe).
    assert finding['issue'].startswith('missing ')
    listed = set(finding['issue'].removeprefix('missing ').split(', '))
    assert 'swe' not in listed
    assert listed == set(dataset.spec.variables) - {'swe'}


def test_completeness_report_respects_date_window(dataset, swe_cog):
    # The only date (2018-04-27) is outside this window, so nothing is reported.
    assert diagnostics.completeness_report(dataset, start=date(2019, 1, 1)) == []


def test_completeness_report_flags_duplicated_cog(dataset, swe_cog):
    # A second file matching swe's glob makes swe *unresolved* (ambiguous) on this
    # date -- the report must list it as an incomplete-date finding rather than
    # crash on the duplicate. A clean date reports nothing.
    duplicate = swe_cog.with_name('us_ssmv11034SlL00T0001TTNATS2018042705HP001-dup.tif')
    shutil.copyfile(swe_cog, duplicate)
    write_swe_cog(dataset, '20180415')  # a clean date alongside the corrupt one

    findings = diagnostics.completeness_report(dataset)

    corrupt = next(f for f in findings if f['target'] == '2018-04-27')
    # duplicated -> unresolved, so swe is listed among the missing variables
    assert 'swe' in corrupt['issue'].removeprefix('missing ').split(', ')
    clean = next(f for f in findings if f['target'] == '2018-04-15')
    # the other date still reports normally (swe present, so not listed)
    assert 'swe' not in clean['issue'].removeprefix('missing ').split(', ')


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
    # A provider expecting a newer format version than what is stamped on the
    # built terrain set is flagged for a rebuild.
    terrain = dataset.zones['terrain']
    stamped = terrain.stored_format_version()
    terrain.format_version = stamped + 1

    findings = diagnostics.stale_format_zone_layers(dataset)

    assert len(findings) == 1
    finding = findings[0]
    assert finding['check'] == 'files'
    assert finding['target'] == 'terrain'
    assert finding['issue'] == (
        f'stale zone-layer format (stored {stamped} != current {stamped + 1})'
    )


def test_stale_format_zone_layers_skips_unbuilt_sets(dataset):
    # An unbuilt set is a missing-artifact finding, not a stale-format one.

    dataset.zones['landcover'].layer_path(FOREST_COVER).unlink()

    targets = {f['target'] for f in diagnostics.stale_format_zone_layers(dataset)}
    assert 'landcover' not in targets


def test_stale_format_zone_layers_flags_a_built_untagged_set(dataset):
    # A built set whose provenance tag is missing (a legacy/untagged build) reads
    # as stale, not unbuilt: gate on presence, not the tag. Rewrite the built
    # terrain COG with no hash tag and it must appear in the stale report with
    # stored=None.
    terrain = dataset.zones['terrain']
    path = terrain.layer_path(ELEVATION)
    with rasterio.open(path) as ds:
        array = ds.read(1)
        profile = ds.profile
        transform = ds.transform
        crs = ds.crs
    write_cog(
        path,
        array,
        transform=transform,
        crs=crs,
        nodata=profile.get('nodata'),
        tile_size=profile['blockxsize'],
        tags={},
    )
    assert terrain.provenance_hash() is None

    findings = diagnostics.stale_format_zone_layers(dataset)

    assert len(findings) == 1
    finding = findings[0]
    assert finding['target'] == 'terrain'
    # A missing/legacy tag reads as stored=None.
    assert finding['issue'] == (
        f'stale zone-layer format (stored None != current {terrain.format_version})'
    )


# --- pourpoint-coverage ------------------------------------------------------------


def _targets_by_issue(findings, issue):
    """Sorted targets of the coverage findings carrying ``issue``."""
    return tuple(sorted(f['target'] for f in findings if f['issue'] == issue))


def test_pourpoint_coverage_unrasterized_then_covered(created_db, pourpoint_geojson):
    db, ds = created_db
    shutil.copy(
        pourpoint_geojson,
        db.pourpoint_records_path / '12345_MT_USGS.geojson',
    )

    before = diagnostics.pourpoint_coverage_report(db, ds)
    assert _targets_by_issue(before, 'no raster') == ('12345:MT:USGS',)
    assert _targets_by_issue(before, 'orphan raster') == ()

    ds.rasterize_aoi(Pourpoint.from_geojson(pourpoint_geojson))
    after = diagnostics.pourpoint_coverage_report(db, ds)
    assert _targets_by_issue(after, 'no raster') == ()


def test_pourpoint_coverage_flags_orphan_raster(created_db, pourpoint_geojson):
    db, ds = created_db  # no global AOIs
    ds.rasterize_aoi(Pourpoint.from_geojson(pourpoint_geojson))

    result = diagnostics.pourpoint_coverage_report(db, ds)

    assert _targets_by_issue(result, 'orphan raster') == ('12345:MT:USGS',)


def test_pourpoint_coverage_classifies_full_partial_none(created_db):
    # The synthetic grid spans lon [-120, -114.88], lat [39.88, 45].
    db, _ds = created_db
    records = db.pourpoint_records_path
    # Fully inside.
    _write_basin(records, '100:MT:USGS', x0=-119.9, y0=44.9, x1=-119.0, y1=44.0)
    # Straddles the western edge -> partial.
    _write_basin(records, '200:MT:USGS', x0=-120.5, y0=44.9, x1=-119.5, y1=44.0)
    # Entirely east of the grid -> none.
    _write_basin(records, '300:MT:USGS', x0=-110.0, y0=44.9, x1=-109.0, y1=44.0)

    result = diagnostics.pourpoint_coverage_report(db, db.datasets['test'])

    assert _targets_by_issue(result, 'partial coverage') == ('200:MT:USGS',)
    assert _targets_by_issue(result, 'no coverage') == ('300:MT:USGS',)


# --- query guard: SnowDb.require_pourpoint_coverage --------------------------------


@pytest.fixture
def guard_db(created_db):
    """A SnowDb with three AOIs: full, partial, and uncovered by the grid."""
    db, _ds = created_db
    records = db.pourpoint_records_path
    _write_basin(records, 'full:MT:USGS', x0=-119.9, y0=44.9, x1=-119.0, y1=44.0)
    _write_basin(records, 'part:MT:USGS', x0=-120.5, y0=44.9, x1=-119.5, y1=44.0)
    _write_basin(records, 'none:MT:USGS', x0=-110.0, y0=44.9, x1=-109.0, y1=44.0)
    # The coverage guard loads pourpoints through the index (availability gate),
    # so the records have to be indexed to be queryable.
    SnowDbManager(db).pourpoints.reindex()
    return db


@pytest.mark.parametrize(
    ('triplet', 'allow_partial', 'expectation'),
    [
        pytest.param('full:MT:USGS', False, Coverage.FULL, id='full-passes'),
        pytest.param('part:MT:USGS', False, 'partially covered', id='partial-raises'),
        pytest.param(
            'part:MT:USGS',
            True,
            Coverage.PARTIAL,
            id='partial-allow-partial-bypasses',
        ),
        pytest.param(
            'none:MT:USGS',
            True,
            'not covered',
            id='uncovered-raises-despite-allow-partial',
        ),
    ],
)
def test_guard_wiring(guard_db, triplet, allow_partial, expectation):
    # Proves the db-level wiring (index -> coverage -> guard); the FULL/PARTIAL/
    # NONE/allow_partial matrix itself is pinned by test_coverage.py's
    # require_full_coverage tests. ``expectation`` is either the Coverage the
    # call must return, or a match string for the PourpointCoverageError it must
    # raise.
    def call():
        return guard_db.require_pourpoint_coverage(
            triplet,
            'test',
            allow_partial=allow_partial,
        )

    if isinstance(expectation, Coverage):
        assert call() is expectation
    else:
        with pytest.raises(PourpointCoverageError, match=expectation):
            call()


# --- pourpoints check: dedup + per-target roll-up ----------------------------


def _write_empty_aoi(dataset, grid, triplet, *, record_path=None):
    """Write an all-zero (but readable) AOI raster for ``triplet``.

    When ``record_path`` (the triplet's stored basin record) is given, stamps
    ``SNOWTOOL_AOI_HASH`` to match it so this hand-written stray/empty raster
    reads as *current*, not stale -- these fixtures are about emptiness, not
    freshness, which is covered separately.
    """
    dataset._aoi_rasters.mkdir(parents=True, exist_ok=True)
    tags = {TILE_BBOX_TAG: '0 0 0 0'}
    if record_path is not None:
        pourpoint = Pourpoint.from_basin_record(record_path)
        tags[AOI_HASH_TAG] = aoi_provenance(
            pourpoint.geometry_hash,
            dataset.nodata_mask_hash,
        )
    write_cog(
        dataset._aoi_rasters / f'{triplet.replace(":", "_")}.tif',
        numpy.zeros((TILE, TILE), dtype=numpy.float32),
        transform=grid.base_grid[0, 0].transform,
        tile_size=TILE,
        nodata=0,
        tags=tags,
        compute_stats=False,
    )


def test_pourpoints_check_reports_stray_raster_for_uncovered_basin(created_db, grid):
    # An off-grid basin cannot be rasterized (`rasterize_aoi` refuses it; the batch
    # path skips it), so an all-zero raster on disk is a stray artifact that should
    # not exist -- a genuine anomaly, not a redundant symptom. It is reported
    # (`empty AOI raster`) alongside the root-cause `no coverage`, rolled onto one
    # row in coverage-then-health order.
    db, ds = created_db
    record_path = _write_basin(
        db.pourpoint_records_path,
        '300:MT:USGS',
        x0=-110.0,
        y0=44.9,
        x1=-109.0,
        y1=44.0,
    )
    _write_empty_aoi(ds, grid, '300:MT:USGS', record_path=record_path)

    findings = diagnostics.run_health_checks(db, [ds], ['pourpoints'])

    for_300 = [f for f in findings if f['target'] == '300:MT:USGS']
    assert len(for_300) == 1
    assert for_300[0]['issue'] == (
        'no coverage; empty AOI raster (covers no in-grid cells: off-grid or masked)'
    )


def test_pourpoints_check_suppresses_no_raster_for_uncovered_basin(created_db):
    # An off-grid basin cannot be rasterized (it has no in-grid pixels), so a
    # missing raster is expected, not a fault. Only the root-cause `no coverage`
    # is reported, not a redundant `no raster` -- for `partial coverage`, though,
    # a raster is expected, so `no raster` still stands (see the roll-up test).
    db, ds = created_db
    _write_basin(
        db.pourpoint_records_path,
        '300:MT:USGS',
        x0=-110.0,
        y0=44.9,
        x1=-109.0,
        y1=44.0,
    )

    findings = diagnostics.run_health_checks(db, [ds], ['pourpoints'])

    for_300 = [f for f in findings if f['target'] == '300:MT:USGS']
    assert len(for_300) == 1
    assert for_300[0]['issue'] == 'no coverage'


def test_pourpoints_check_keeps_empty_aoi_for_covered_basin(created_db, grid):
    # A covered basin whose raster is empty (e.g. entirely masked) is a genuine
    # fault: `empty AOI raster` is reported here just as it is for the off-grid
    # stray-raster case -- an all-zero raster always surfaces, covered or not.
    db, ds = created_db
    record_path = _write_basin(
        db.pourpoint_records_path,
        '100:MT:USGS',
        x0=-119.9,
        y0=44.9,
        x1=-119.0,
        y1=44.0,
    )
    _write_empty_aoi(ds, grid, '100:MT:USGS', record_path=record_path)

    findings = diagnostics.run_health_checks(db, [ds], ['pourpoints'])

    for_100 = [f for f in findings if f['target'] == '100:MT:USGS']
    assert len(for_100) == 1
    assert for_100[0]['issue'].startswith('empty AOI')


def test_pourpoints_check_rolls_up_issues_for_one_target(created_db):
    # A partial-coverage basin with no burned raster hits two pourpoint issues
    # (`no raster` + `partial coverage`); doctor collapses them onto one row so a
    # target never spans multiple lines. Emission order is coverage-report order.
    db, ds = created_db
    _write_basin(
        db.pourpoint_records_path,
        '200:MT:USGS',
        x0=-120.5,
        y0=44.9,
        x1=-119.5,
        y1=44.0,
    )

    findings = diagnostics.run_health_checks(db, [ds], ['pourpoints'])

    for_200 = [f for f in findings if f['target'] == '200:MT:USGS']
    assert len(for_200) == 1
    assert for_200[0]['issue'] == 'no raster; partial coverage'


def test_doctor_pourpoints_reports_stale_aoi_hash(created_db, pourpoint_geojson):
    # Doctor now routes AOI validation through `aoi_raster_issues`, so a raster
    # whose stamped SNOWTOOL_AOI_HASH no longer matches its basin record's
    # current geometry hash (a basin edited out from under an already-rasterized
    # pourpoint) is reported -- not just structural/emptiness faults.
    db, ds = created_db
    shutil.copy(
        pourpoint_geojson,
        db.pourpoint_records_path / '12345_MT_USGS.geojson',
    )
    pp = Pourpoint.from_geojson(pourpoint_geojson)
    ds.rasterize_aoi(pp)
    path = ds.aoi_raster_path_from_triplet(pp.station_triplet)

    # Re-stamp the raster's AOI-hash tag with a bogus value, simulating drift
    # between the stored record and the burned raster's provenance.
    with rasterio.open(path) as src:
        array = src.read(1)
        transform = src.transform
        crs = src.crs
        tags = dict(src.tags())
    tags[AOI_HASH_TAG] = 'v1:deadbeef'
    write_cog(
        path,
        array,
        transform=transform,
        crs=crs,
        nodata=AOI_MASK_NODATA,
        tile_size=TILE,
        tags=tags,
        compute_stats=False,
    )

    findings = diagnostics.run_health_checks(db, [ds], ['pourpoints'])

    for_pp = [f for f in findings if f['target'] == pp.station_triplet]
    assert len(for_pp) == 1
    assert 'stale content' in for_pp[0]['issue']


def test_doctor_pourpoints_no_spurious_findings_for_offset_basin(created_db, tmp_path):
    # A basin whose tile-bbox does not start at tile (0, 0) must not spuriously
    # trip the AOI-validation step now that it also checks grid-compat +
    # freshness through `aoi_raster_issues` -- a normal, correctly-rasterized
    # offset basin still reports nothing for `pourpoints`.
    db, ds = created_db
    record_path = _write_basin(
        db.pourpoint_records_path,
        '55555:MT:USGS',
        x0=-115.7,
        y0=40.0,
        x1=-115.3,
        y1=40.4,
    )
    pp = Pourpoint.from_geojson(record_path)
    ds.rasterize_aoi(pp)

    findings = diagnostics.run_health_checks(db, [ds], ['pourpoints'])

    assert [f for f in findings if f['target'] == '55555:MT:USGS'] == []


# --- doctor progress: per-item enumeration + live labels ---------------------


class _RecordingTask:
    def __init__(self, rec):
        self._rec = rec

    def advance(self, n=1):
        self._rec.advances += n

    def describe(self, label):
        self._rec.descriptions.append(label)


class RecordingProgress:
    """A ProgressReporter that records the total and every describe/advance."""

    def __init__(self):
        self.total = None
        self.advances = 0
        self.descriptions = []

    @contextlib.contextmanager
    def track(self, label, *, total=None):
        self.total = total
        yield _RecordingTask(self)


def test_run_health_checks_reports_one_step_per_raster_and_check(
    created_db,
    pourpoint_geojson,
):
    # One increment per unit of work (per COG for grid, per AOI raster for
    # pourpoints, plus the atomic declaration/inventory steps), all enumerated
    # up front so `total` is exact, and each step announces what it is doing.
    db, ds = created_db
    write_swe_cog(ds, '20180427')  # one COG
    shutil.copy(pourpoint_geojson, db.pourpoint_records_path / '12345_MT_USGS.geojson')
    ds.rasterize_aoi(Pourpoint.from_geojson(pourpoint_geojson))  # one AOI raster

    rec = RecordingProgress()
    diagnostics.run_health_checks(db, [ds], ['grid', 'pourpoints'], progress=rec)

    # grid: declaration + 1 COG; pourpoints: per-basin coverage (1) + orphan
    # rasters + 1 AOI raster. Coverage is one step *per pourpoint*, so a large
    # registry advances the bar instead of stalling on a single inventory step.
    assert rec.total == 5
    assert rec.advances == 5
    assert len(rec.descriptions) == 5
    assert rec.descriptions[0] == 'test grid: declaration'
    assert rec.descriptions[1].startswith('test grid: 20180427/')
    assert rec.descriptions[2] == 'test pourpoints: coverage 12345:MT:USGS'
    assert rec.descriptions[3] == 'test pourpoints: orphan rasters'
    assert rec.descriptions[4] == 'test pourpoints: AOI validation 12345:MT:USGS'


# --- aoi-health --------------------------------------------------------------


def test_aoi_health_all_healthy(dataset, pourpoint_geojson):
    dataset.rasterize_aoi(Pourpoint.from_geojson(pourpoint_geojson))

    health = diagnostics.aoi_health_report(dataset)

    assert health == []


def test_aoi_health_flags_empty_aoi(dataset, grid):
    # AOI rasters carry per-pixel cell area (decoupled from the DEM). An
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
    assert bad[0]['check'] == 'pourpoints'
    assert 'empty AOI' in bad[0]['issue']


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
    assert any('TILE_BBOX' in h['issue'] for h in bad)


def test_aoi_raster_issues_clean_for_current_raster(dataset, pourpoint_geojson):
    pp = Pourpoint.from_geojson(pourpoint_geojson)
    dataset.rasterize_aoi(pp)
    path = dataset.aoi_raster_path_from_triplet(pp.station_triplet)

    result = aoi_raster_issues(
        path,
        grid=dataset.grid,
        expected_hash=aoi_provenance(pp.geometry_hash, dataset.nodata_mask_hash),
    )

    assert result == []


def test_aoi_raster_issues_clean_for_offset_basin(dataset, tmp_path):
    # False-positive guard: a basin whose tile-bbox does NOT start at tile (0, 0)
    # must still be clean. Its UL-tile transform is offset from the grid origin,
    # so comparing the raster's own transform (not the grid base transform) is
    # what keeps this from spuriously flagging GridMismatch. The synthetic grid
    # spans lon [-120, -114.88], lat [39.88, 45]; a box around (-115.5, 40.2)
    # lands in tile (1, 1).
    record = _write_basin(
        tmp_path,
        '55555:MT:USGS',
        x0=-115.7,
        y0=40.0,
        x1=-115.3,
        y1=40.4,
    )
    pp = Pourpoint.from_geojson(record)
    dataset.rasterize_aoi(pp)
    path = dataset.aoi_raster_path_from_triplet(pp.station_triplet)

    result = aoi_raster_issues(
        path,
        grid=dataset.grid,
        expected_hash=aoi_provenance(pp.geometry_hash, dataset.nodata_mask_hash),
    )

    assert result == []


def test_aoi_raster_issues_flags_grid_drift(dataset, pourpoint_geojson):
    # True-positive: the SAME on-disk raster, read against a grid whose origin is
    # shifted by one pixel, must be flagged -- proving real grid drift (not a
    # value re-derived from the current grid) is detected.
    pp = Pourpoint.from_geojson(pourpoint_geojson)
    dataset.rasterize_aoi(pp)
    path = dataset.aoi_raster_path_from_triplet(pp.station_triplet)

    shifted_grid = synthetic_grid(origin_x=ORIGIN_X + PX)

    result = aoi_raster_issues(
        path,
        grid=shifted_grid,
        expected_hash=aoi_provenance(pp.geometry_hash, dataset.nodata_mask_hash),
    )

    assert any(isinstance(i, issues_mod.GridMismatch) for i in result)
    assert all(i.actionable for i in result if isinstance(i, issues_mod.GridMismatch))


def test_aoi_raster_issues_flags_stale_hash(dataset, pourpoint_geojson):
    pp = Pourpoint.from_geojson(pourpoint_geojson)
    dataset.rasterize_aoi(pp)
    path = dataset.aoi_raster_path_from_triplet(pp.station_triplet)

    result = aoi_raster_issues(
        path,
        grid=dataset.grid,
        expected_hash='v1:deadbeef',  # deliberately not what was stamped
    )

    assert any(isinstance(i, issues_mod.ContentStale) for i in result)
    assert all(i.actionable for i in result if isinstance(i, issues_mod.ContentStale))


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
    assert result.status.present is True
    assert result.grid.is_geographic is True
    assert result.grid.cell_area_m2 is None  # geographic grid -> per-pixel area raster
    assert result.grid.rows == result.grid.cols == 512
    assert result.grid.n_tiles == 4
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
    assert result.status.artifacts.cogs is False  # no dates ingested
    assert result.status.artifacts.aoi_rasters is False
    assert result.status.date_count == 0
    assert result.status.first_date is None
    assert result.status.last_date is None


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


def test_grid_check_validates_every_cog_not_just_the_first(created_db):
    # A clean COG on an early date and a drifted COG on a later date: the grid
    # check opens *every* header, so the later drift is caught even though the
    # first COG is fine (a first-COG-only sample would miss it). The finding names
    # the specific file so it is actionable.
    db, ds = created_db
    write_swe_cog(ds, '20180101')  # aligned to the declared grid -> clean
    bad_dir = ds._cogs / '20180201'
    bad_dir.mkdir(parents=True)
    write_cog(
        bad_dir / f'{snodas_swe_name("20180201")}.tif',
        numpy.zeros((256, 256), dtype=numpy.int16),
        transform=ds.grid.base_grid.transform,
        tile_size=TILE,
    )

    findings = diagnostics.run_health_checks(db, [ds], ['grid'])

    grid = [f for f in findings if f['check'] == 'grid']
    assert len(grid) == 1
    assert grid[0]['target'] == f'20180201/{snodas_swe_name("20180201")}.tif'
    assert '256x256' in grid[0]['issue']


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
