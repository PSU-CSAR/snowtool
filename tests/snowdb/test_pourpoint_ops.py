"""SnowDb AOI import/sync/dump/remove + Dataset staleness/cascade helpers."""

import json

import numpy
import pytest

import snowtool.snowdb.aoi_raster as aoi_raster_mod

from snowtool.exceptions import (
    GeoJSONValidationError,
    GeometryOutsideGridError,
    IndexedPourpointMissingBasinError,
    PourpointNotFoundError,
    PourpointPruneDestinationRequiredError,
)
from snowtool.snowdb.aoi_raster import aoi_provenance
from snowtool.snowdb.coverage import Coverage
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.manager import SnowDbManager
from snowtool.snowdb.pourpoint import Pourpoint
from snowtool.snowdb.pourpoint_manager import PourpointManager

from ..conftest import (
    CapturingProgress,
    make_manager,
    make_spec,
)
from ..conftest import write_aoi_record as _write_aoi

_POINT = {'type': 'Point', 'coordinates': [-119.45, 44.45]}


def _box(x0=-119.9, y0=44.9, x1=-119.0, y1=44.0):
    """A rectangular Polygon geometry inside the synthetic grid's first tile."""
    return {
        'type': 'Polygon',
        'coordinates': [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }


# A polygon well inside the synthetic grid's first tile (see top-level conftest).
_POLYGON = _box()


@pytest.fixture
def manager(tmp_path, spec):
    """An initialized snowdb (write side) with the synthetic 'test' dataset.

    Writes go through this manager; reads use the derived ``db`` fixture.
    """
    SnowDbManager.initialize(tmp_path)
    Dataset.create(spec, tmp_path / 'data' / 'test')  # skeleton only; return unused
    return make_manager(tmp_path, [spec])


@pytest.fixture
def db(manager):
    """The manager's read SnowDb (same underlying root)."""
    return manager.db


# --- import ------------------------------------------------------------------


def test_import_file_writes_record_and_index(manager, db, pourpoint_geojson):
    result = manager.pourpoints.import_(pourpoint_geojson)

    assert result.imported == ['12345:MT:USGS']
    assert db.pourpoint_record_path('12345:MT:USGS').is_file()
    assert db.pourpoint_index().triplets() == {'12345:MT:USGS'}


def test_import_dir_classifies_imported_skipped_invalid(manager, tmp_path):
    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')
    _write_aoi(src, '22222:MT:USGS', with_polygon=False)  # point-only -> skipped
    (src / 'bad.geojson').write_text(json.dumps({'type': 'Nonsense'}))

    result = manager.pourpoints.import_(src)

    assert result.imported == ['11111:MT:USGS']
    assert result.skipped == ['22222:MT:USGS']
    assert [p.name for p, _ in result.invalid] == ['bad.geojson']
    assert manager.db.pourpoint_triplets() == {'11111:MT:USGS'}


def test_import_dry_run_writes_nothing(manager, db, pourpoint_geojson):
    result = manager.pourpoints.import_(pourpoint_geojson, dry_run=True)

    assert result.imported == ['12345:MT:USGS']
    assert not db.pourpoint_record_path('12345:MT:USGS').exists()
    assert not db.pourpoint_index_path.exists()


def test_import_is_idempotent(manager, pourpoint_geojson):
    manager.pourpoints.import_(pourpoint_geojson)
    manager.pourpoints.import_(pourpoint_geojson)

    assert manager.db.pourpoint_triplets() == {'12345:MT:USGS'}


# --- malformed source classification (3a) ------------------------------------


def _write_bad(path, kind):
    """Write a source file that is unreadable as a pourpoint, three ways."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == 'nonjson':
        path.write_text('this is not json {{{')
    elif kind == 'nonutf8':
        path.write_bytes(b'\xff\xfe\x00 not utf-8 bytes')
    elif kind == 'wrong_shape':
        # Valid JSON, but not a GeoJSON object (a bare list).
        path.write_text(json.dumps([1, 2, 3]))
    else:  # pragma: no cover - test helper guard
        raise ValueError(kind)


@pytest.mark.parametrize('kind', ['nonjson', 'nonutf8', 'wrong_shape'])
def test_from_geojson_maps_malformed_to_validation_error(tmp_path, kind):
    # Decode/parse failures must surface as GeoJSONValidationError (the error
    # _classify_sources catches), not a raw JSONDecodeError/UnicodeDecodeError.
    bad = tmp_path / 'bad.geojson'
    _write_bad(bad, kind)

    with pytest.raises(GeoJSONValidationError):
        Pourpoint.from_geojson(bad)


@pytest.mark.parametrize('kind', ['nonjson', 'nonutf8', 'wrong_shape'])
@pytest.mark.parametrize('dry_run', [False, True])
def test_import_malformed_source_lands_in_invalid(manager, tmp_path, kind, dry_run):
    # A single garbage file must not abort the whole batch: the good record still
    # imports and the bad one is classified as invalid (both dry-run and real).
    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')
    _write_bad(src / 'bad.geojson', kind)

    result = manager.pourpoints.import_(src, dry_run=dry_run)

    assert result.imported == ['11111:MT:USGS']
    assert [p.name for p, _ in result.invalid] == ['bad.geojson']


@pytest.mark.parametrize('kind', ['nonjson', 'nonutf8', 'wrong_shape'])
@pytest.mark.parametrize('dry_run', [False, True])
def test_sync_malformed_source_lands_in_invalid(manager, tmp_path, kind, dry_run):
    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')
    _write_bad(src / 'bad.geojson', kind)

    result = manager.pourpoints.sync(src, dry_run=dry_run)

    assert result.imported == ['11111:MT:USGS']
    assert [p.name for p, _ in result.invalid] == ['bad.geojson']


def _source(**overrides):
    """A valid two-geometry pourpoint source, with per-test overrides."""
    return {
        'type': 'GeometryCollection',
        'id': '12345:MT:USGS',
        'geometries': [_POINT, _POLYGON],
        'properties': {'name': 'Test Basin'},
        **overrides,
    }


@pytest.mark.parametrize(
    'source',
    [
        pytest.param(_source(id='not a triplet'), id='malformed-triplet'),
        pytest.param(_source(id=12345), id='non-string-id'),
        pytest.param(_source(geometries=[_POINT]), id='one-geometry'),
        pytest.param(
            _source(geometries=[_POINT, _POINT, _POLYGON]),
            id='three-geometries',
        ),
        pytest.param(_source(geometries=[_POINT, _POINT]), id='two-points'),
        pytest.param(_source(geometries=[_POLYGON, _POLYGON]), id='two-polygons'),
        pytest.param(_source(properties=None), id='null-properties'),
        pytest.param(_source(properties={'source': 'test'}), id='no-name'),
        pytest.param(_source(type='FeatureCollection'), id='unsupported-type'),
        pytest.param(
            {
                'type': 'Feature',
                'id': '12345:MT:USGS',
                'geometry': _POLYGON,
                'properties': {'name': 'Test Basin'},
            },
            id='feature-with-polygon-geometry',
        ),
    ],
)
def test_from_geojson_rejects_structurally_invalid_sources(tmp_path, source):
    # Every structural defect -- a bad triplet id, the wrong geometry count or
    # kinds, null properties (which previously crashed the batch with an
    # AttributeError), a missing name -- classifies as GeoJSONValidationError.
    path = tmp_path / 'bad.geojson'
    path.write_text(json.dumps(source))
    with pytest.raises(GeoJSONValidationError):
        Pourpoint.from_geojson(path)


def test_from_geojson_parses_a_point_only_feature(tmp_path):
    src = {
        'type': 'Feature',
        'id': '99999:MT:USGS',
        'geometry': _POINT,
        'properties': {'name': 'Point Only', 'awdb_id': '99999'},
    }
    path = tmp_path / 'point.geojson'
    path.write_text(json.dumps(src))
    pp = Pourpoint.from_geojson(path)
    assert pp.station_triplet == '99999:MT:USGS'
    assert pp.polygon is None
    assert tuple(pp.point.coordinates[:2]) == (-119.45, 44.45)
    assert pp.name == 'Point Only'
    assert pp.awdb_id == '99999'
    assert pp.properties == {'name': 'Point Only', 'awdb_id': '99999'}


def test_from_geojson_prefers_nwccname_over_name(tmp_path):
    src = _source(properties={'nwccname': 'NWCC Name', 'name': 'Other'})
    path = tmp_path / 'pp.geojson'
    path.write_text(json.dumps(src))
    assert Pourpoint.from_geojson(path).name == 'NWCC Name'


# --- sync --------------------------------------------------------------------


def test_sync_prunes_absent_aoi_and_cascades(manager, db, tmp_path):
    # Two stored AOIs; sync a source dir that only has one.
    manager.pourpoints.import_(_write_aoi(tmp_path / 'seed', '11111:MT:USGS').parent)
    manager.pourpoints.import_(_write_aoi(tmp_path / 'seed', '22222:MT:USGS').parent)
    # Burn a raster for the one about to be pruned, to prove the cascade.
    db['test'].rasterize_aoi(db.load_pourpoint('22222:MT:USGS'))
    raster = db['test'].aoi_raster_path_from_triplet('22222:MT:USGS')
    assert raster.is_file()

    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')
    archive = tmp_path / 'archive'

    result = manager.pourpoints.sync(src, prune_to=archive)

    assert result.pruned == ['22222:MT:USGS']
    assert not db.pourpoint_record_path('22222:MT:USGS').exists()
    assert not raster.exists()  # cascade
    assert (archive / '22222_MT_USGS.geojson').is_file()  # dumped first
    assert db.pourpoint_index().triplets() == {'11111:MT:USGS'}


def test_sync_without_prune_to_refuses_to_remove(manager, db, tmp_path):
    manager.pourpoints.import_(_write_aoi(tmp_path / 'seed', '22222:MT:USGS').parent)
    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')

    with pytest.raises(PourpointPruneDestinationRequiredError):
        manager.pourpoints.sync(src)

    # Nothing was changed (the additive import did not run either).
    assert manager.db.pourpoint_triplets() == {'22222:MT:USGS'}
    assert not db.pourpoint_record_path('11111:MT:USGS').exists()


def test_sync_dry_run_reports_prune_without_removing(manager, tmp_path):
    manager.pourpoints.import_(_write_aoi(tmp_path / 'seed', '22222:MT:USGS').parent)
    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')

    result = manager.pourpoints.sync(src, dry_run=True)

    assert result.pruned == ['22222:MT:USGS']
    assert manager.db.pourpoint_triplets() == {'22222:MT:USGS'}  # unchanged


# --- dump / remove / reindex -------------------------------------------------


def test_dump_aoi_copies_record_out(manager, db, pourpoint_geojson, tmp_path):
    manager.pourpoints.import_(pourpoint_geojson)
    dest = db.dump_pourpoint('12345:MT:USGS', tmp_path / 'out')
    assert dest == tmp_path / 'out' / '12345_MT_USGS.geojson'
    assert dest.is_file()


def test_remove_aoi_cascades_and_reindexes(manager, db, pourpoint_geojson):
    manager.pourpoints.import_(pourpoint_geojson)
    db['test'].rasterize_aoi(db.load_pourpoint('12345:MT:USGS'))
    raster = db['test'].aoi_raster_path_from_triplet('12345:MT:USGS')
    assert raster.is_file()

    assert manager.pourpoints.remove('12345:MT:USGS') is True
    assert not db.pourpoint_record_path('12345:MT:USGS').exists()
    assert not raster.exists()
    assert db.pourpoint_index().triplets() == set()


def test_remove_absent_aoi_is_a_noop(manager):
    assert manager.pourpoints.remove('99999:MT:USGS') is False


def test_reindex_rebuilds_from_records(manager, db, pourpoint_geojson):
    manager.pourpoints.import_(pourpoint_geojson)
    db.pourpoint_index_path.unlink()

    index = manager.pourpoints.reindex()
    assert index.triplets() == {'12345:MT:USGS'}
    assert db.pourpoint_index_path.is_file()


def test_reindex_raises_on_basin_less_stored_record(manager, db, pourpoint_geojson):
    # Every stored record is basin-bearing (the import boundary guarantees it), so
    # a point-only record in `records/` is a corrupt store. Reindex refuses loudly
    # -- naming the offending file -- rather than silently dropping the record.
    manager.pourpoints.import_(pourpoint_geojson)
    triplet = '12345:MT:USGS'
    record_path = db.pourpoint_record_path(triplet)
    _write_aoi(
        db.pourpoint_records_path,
        triplet,
        with_polygon=False,
    )

    with pytest.raises(
        IndexedPourpointMissingBasinError,
        match=record_path.name,
    ):
        manager.pourpoints.reindex()


def test_basin_pourpoints_raises_on_basin_less_record(manager, db, pourpoint_geojson):
    # basins() enforces the basin-bearing invariant on read: a point-only
    # record edited into `records/` out of band raises the typed error naming the
    # file, rather than flowing on to fail with an untyped ValueError downstream.
    manager.pourpoints.import_(pourpoint_geojson)
    triplet = '12345:MT:USGS'
    record_path = db.pourpoint_record_path(triplet)
    _write_aoi(
        db.pourpoint_records_path,
        triplet,
        with_polygon=False,
    )

    with pytest.raises(
        IndexedPourpointMissingBasinError,
        match=record_path.name,
    ):
        PourpointManager(db).basins()


def test_load_pourpoint_gates_on_index(manager, db):
    """A record dropped into ``records/`` out of band (never reindexed) is not
    served: the index is the availability gate. A reindex makes it loadable."""
    _write_aoi(db.pourpoint_records_path, '77777:MT:USGS')
    assert db.pourpoint_record_path('77777:MT:USGS').is_file()

    with pytest.raises(PourpointNotFoundError):
        db.load_pourpoint('77777:MT:USGS')

    manager.pourpoints.reindex()
    assert db.load_pourpoint('77777:MT:USGS').station_triplet == '77777:MT:USGS'


# --- incremental index maintenance --------------------------------------------


def _mutate_record_name(db, triplet, name):
    """Edit a stored record's ``name`` out of band (no reindex): index drift."""
    path = db.pourpoint_record_path(triplet)
    record = json.loads(path.read_text())
    record['properties']['name'] = name
    path.write_text(json.dumps(record))


def test_import_reuses_untouched_index_entries(manager, db, tmp_path):
    manager.pourpoints.import_(_write_aoi(tmp_path / 'seed', '11111:MT:USGS').parent)
    _mutate_record_name(db, '11111:MT:USGS', 'MUTATED')

    # Importing a *different* pourpoint must not re-parse the untouched record:
    # its entry is reused as-is, so the out-of-band drift stays invisible.
    manager.pourpoints.import_(_write_aoi(tmp_path / 'new', '22222:MT:USGS').parent)

    index = db.pourpoint_index()
    assert index.triplets() == {'11111:MT:USGS', '22222:MT:USGS'}
    assert index['11111:MT:USGS'].name == '11111:MT:USGS'  # reused (stale name)
    assert index['22222:MT:USGS'].name == '22222:MT:USGS'

    # The explicit full rebuild is the recovery path for out-of-band edits.
    manager.pourpoints.reindex()
    assert db.pourpoint_index()['11111:MT:USGS'].name == 'MUTATED'


def test_remove_drops_removed_entry_and_reuses_the_rest(manager, db, tmp_path):
    manager.pourpoints.import_(_write_aoi(tmp_path / 'seed', '11111:MT:USGS').parent)
    manager.pourpoints.import_(_write_aoi(tmp_path / 'seed', '22222:MT:USGS').parent)
    _mutate_record_name(db, '11111:MT:USGS', 'MUTATED')

    assert manager.pourpoints.remove('22222:MT:USGS') is True

    index = db.pourpoint_index()
    assert index.triplets() == {'11111:MT:USGS'}
    # Still the pre-mutation name: the surviving entry was reused, not re-parsed.
    assert index['11111:MT:USGS'].name == '11111:MT:USGS'


def test_sync_prune_drops_pruned_and_reuses_untouched(manager, db, tmp_path):
    manager.pourpoints.import_(_write_aoi(tmp_path / 'seed', '11111:MT:USGS').parent)
    manager.pourpoints.import_(_write_aoi(tmp_path / 'seed', '22222:MT:USGS').parent)
    _mutate_record_name(db, '11111:MT:USGS', 'MUTATED')

    # 11111 is point-only in the source: present (not pruned) but not
    # re-imported, so its stored record survives and its entry must be reused.
    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS', with_polygon=False)

    result = manager.pourpoints.sync(src, prune_to=tmp_path / 'archive')

    assert result.pruned == ['22222:MT:USGS']
    index = db.pourpoint_index()
    assert index.triplets() == {'11111:MT:USGS'}
    assert index['11111:MT:USGS'].name == '11111:MT:USGS'  # reused (stale name)


def test_new_dataset_registration_defeats_entry_reuse(manager, db, tmp_path, spec):
    manager.pourpoints.import_(_write_aoi(tmp_path / 'seed', '11111:MT:USGS').parent)
    _mutate_record_name(db, '11111:MT:USGS', 'MUTATED')
    assert set(db.pourpoint_index()['11111:MT:USGS'].coverage) == {'test'}

    # A second registered dataset changes the coverage key set, so the reuse
    # guard fails and the entry is rebuilt from disk on the next import --
    # picking up both the new coverage key and the record's current content.
    other = make_spec('other', spec)
    manager2 = make_manager(tmp_path, [spec, other])
    manager2.pourpoints.import_(_write_aoi(tmp_path / 'new', '22222:MT:USGS').parent)

    index = manager2.db.pourpoint_index()
    entry = index['11111:MT:USGS']
    assert set(entry.coverage) == {'test', 'other'}
    assert entry.coverage['other'] is Coverage.FULL
    assert entry.name == 'MUTATED'  # rebuilt from disk, not reused


def test_missing_index_self_heals_on_import(manager, db, tmp_path):
    manager.pourpoints.import_(_write_aoi(tmp_path / 'seed', '11111:MT:USGS').parent)
    db.pourpoint_index_path.unlink()

    manager.pourpoints.import_(_write_aoi(tmp_path / 'new', '22222:MT:USGS').parent)

    # No old entry to reuse -> the fallback re-parses the record from disk.
    assert db.pourpoint_index().triplets() == {'11111:MT:USGS', '22222:MT:USGS'}


# --- progress reporting --------------------------------------------------------


def test_import_reports_parse_and_index_phases(manager, tmp_path):
    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')
    _write_aoi(src, '22222:MT:USGS')
    progress = CapturingProgress()

    manager.pourpoints.import_(src, progress=progress)

    assert [(t.label, t.total, t.advanced) for t in progress.tasks] == [
        ('parsing 2 pourpoint source(s)', 2, 2),
        ('indexing 2 pourpoint(s)', 2, 2),
    ]


def test_import_dry_run_reports_only_the_parse_phase(manager, tmp_path):
    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')
    progress = CapturingProgress()

    manager.pourpoints.import_(src, dry_run=True, progress=progress)

    assert [(t.label, t.total, t.advanced) for t in progress.tasks] == [
        ('parsing 1 pourpoint source(s)', 1, 1),
    ]


def test_sync_reports_parse_and_index_phases(manager, tmp_path):
    manager.pourpoints.import_(_write_aoi(tmp_path / 'seed', '22222:MT:USGS').parent)
    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')
    progress = CapturingProgress()

    manager.pourpoints.sync(src, prune_to=tmp_path / 'archive', progress=progress)

    # The index phase totals the *surviving* records (22222 was pruned).
    assert [(t.label, t.total, t.advanced) for t in progress.tasks] == [
        ('parsing 1 pourpoint source(s)', 1, 1),
        ('indexing 1 pourpoint(s)', 1, 1),
    ]


def test_reindex_reports_the_index_phase(manager, tmp_path):
    manager.pourpoints.import_(_write_aoi(tmp_path / 'seed', '11111:MT:USGS').parent)
    manager.pourpoints.import_(_write_aoi(tmp_path / 'seed', '22222:MT:USGS').parent)
    progress = CapturingProgress()

    manager.pourpoints.reindex(progress=progress)

    assert [(t.label, t.total, t.advanced) for t in progress.tasks] == [
        ('indexing 2 pourpoint(s)', 2, 2),
    ]


def test_remove_reports_the_index_phase(manager, tmp_path):
    manager.pourpoints.import_(_write_aoi(tmp_path / 'seed', '11111:MT:USGS').parent)
    manager.pourpoints.import_(_write_aoi(tmp_path / 'seed', '22222:MT:USGS').parent)
    progress = CapturingProgress()

    manager.pourpoints.remove('22222:MT:USGS', progress=progress)

    assert [(t.label, t.total, t.advanced) for t in progress.tasks] == [
        ('indexing 1 pourpoint(s)', 1, 1),
    ]


# --- Dataset staleness + cascade primitives ----------------------------------


def test_aoi_raster_hash_matches_aoi(db, pourpoint_geojson):

    aoi = Pourpoint.from_geojson(pourpoint_geojson)
    db['test'].rasterize_aoi(aoi)
    # The stored tag is the versioned provenance (geometry hash + format version),
    # not the bare geometry hash.
    assert db['test'].aoi_raster_hash('12345:MT:USGS') == aoi_provenance(
        aoi.geometry_hash,
        None,
    )
    assert db['test'].aoi_raster_is_current(aoi)


def test_aoi_raster_hash_none_when_absent(db):
    assert db['test'].aoi_raster_hash('99999:MT:USGS') is None


def test_format_version_bump_makes_aoi_raster_stale(db, pourpoint_geojson, monkeypatch):
    # A material format change is detected by the same staleness check as a changed
    # basin: bump the burned-raster format version and an already-current raster
    # (unchanged geometry) reads as stale, forcing a rebuild.

    aoi = Pourpoint.from_geojson(pourpoint_geojson)
    ds = db['test']
    ds.rasterize_aoi(aoi)
    assert ds.aoi_raster_is_current(aoi) is True

    monkeypatch.setattr(
        aoi_raster_mod,
        'AOI_RASTER_FORMAT_VERSION',
        aoi_raster_mod.AOI_RASTER_FORMAT_VERSION + 1,
    )
    assert ds.aoi_raster_is_current(aoi) is False
    assert ds.rasterize_aoi(aoi) is True
    assert ds.aoi_raster_is_current(aoi) is True


def test_rasterize_aoi_builds_then_skips_then_rebuilds(
    db,
    pourpoint_geojson,
    tmp_path,
):
    ds = db['test']
    aoi = Pourpoint.from_geojson(pourpoint_geojson)

    assert ds.rasterize_aoi(aoi) is True  # missing -> built
    assert ds.rasterize_aoi(aoi) is False  # current -> skipped

    # A changed basin makes the existing raster stale.
    stale = _write_aoi(
        tmp_path / 'stale',
        '12345:MT:USGS',
        polygon=_box(-119.8, 44.8, -119.1, 44.1),
    )
    stale_aoi = Pourpoint.from_geojson(stale)
    assert ds.aoi_raster_is_current(stale_aoi) is False
    assert ds.rasterize_aoi(stale_aoi) is True  # stale -> rebuilt
    assert ds.rasterize_aoi(stale_aoi, rebuild=False) is False


def test_remove_aoi_raster(db, pourpoint_geojson):
    ds = db['test']
    ds.rasterize_aoi(Pourpoint.from_geojson(pourpoint_geojson))
    assert ds.remove_aoi_raster('12345:MT:USGS') is True
    assert ds.remove_aoi_raster('12345:MT:USGS') is False  # idempotent


def test_rasterize_aois_built_and_skipped(manager, db, pourpoint_geojson):
    aoi = Pourpoint.from_geojson(pourpoint_geojson)
    datasets = list(db.datasets.values())

    first = manager.pourpoints.rasterize_aois([aoi], datasets)
    assert first.built == [('12345:MT:USGS', 'test')]
    assert first.skipped == []

    second = manager.pourpoints.rasterize_aois([aoi], datasets)
    assert second.built == []
    assert second.skipped == [('12345:MT:USGS', 'test')]


# --- basins beyond the grid edge (window clamping + off-grid skips) -----------

# A basin straddling the grid's north edge (45N): lon -119.9..-119.0, lat
# 44.5..45.5 -- the top half spills off the grid.
_STRADDLE = _box(-119.9, 45.5, -119.0, 44.5)
# The same basin pre-clipped to the grid's north edge (the in-grid part).
_STRADDLE_CLIPPED = _box(-119.9, 45.0, -119.0, 44.5)
# A basin entirely north of the grid.
_OUTSIDE = _box(-119.9, 46.9, -119.0, 46.0)


def test_rasterize_straddling_basin_clamps_to_the_grid(db, tmp_path):
    straddle = Pourpoint.from_geojson(
        _write_aoi(tmp_path / 'src', '33333:MT:USGS', polygon=_STRADDLE),
    )

    db['test'].rasterize_aoi(straddle)
    raster = db['test'].load_aoi_raster(straddle.station_triplet)

    # The window is clamped to the single in-grid tile (the bug produced an
    # inverted window here: griffine wrapped the negative tile row to the last).
    assert [(t.row, t.col) for t in raster.tiles] == [(0, 0)]
    # Only the in-grid part burns: 90 cols (lon -119.9..-119.0) x 50 rows
    # (lat 45.0..44.5) of 0.01-degree pixels.
    assert (raster.array > 0).sum() == 90 * 50
    # And it burns *identically* to the polygon pre-clipped to the grid edge.
    clipped = Pourpoint.from_geojson(
        _write_aoi(tmp_path / 'src', '44444:MT:USGS', polygon=_STRADDLE_CLIPPED),
    )
    db['test'].rasterize_aoi(clipped)
    clipped_raster = db['test'].load_aoi_raster(clipped.station_triplet)
    assert numpy.array_equal(raster.array, clipped_raster.array)


def test_rasterize_fully_outside_basin_raises_typed_error(db, tmp_path):
    outside = Pourpoint.from_geojson(
        _write_aoi(tmp_path / 'src', '55555:MT:USGS', polygon=_OUTSIDE),
    )

    with pytest.raises(GeometryOutsideGridError, match='do not intersect'):
        db['test'].rasterize_aoi(outside)

    assert not db['test'].aoi_raster_path_from_triplet('55555:MT:USGS').exists()


def test_rasterize_aois_skips_off_grid_basins(manager, db, tmp_path):
    inside = Pourpoint.from_geojson(_write_aoi(tmp_path / 'src', '11111:MT:USGS'))
    outside = Pourpoint.from_geojson(
        _write_aoi(tmp_path / 'src', '55555:MT:USGS', polygon=_OUTSIDE),
    )

    result = manager.pourpoints.rasterize_aois(
        [inside, outside],
        list(db.datasets.values()),
    )

    assert result.built == [('11111:MT:USGS', 'test')]
    assert result.skipped == [('55555:MT:USGS', 'test')]
    assert db['test'].aoi_raster_path_from_triplet('11111:MT:USGS').is_file()
    assert not db['test'].aoi_raster_path_from_triplet('55555:MT:USGS').exists()


def test_stage_dataset_records_coverage_and_skips_off_grid(manager, tmp_path, spec):
    from snowtool.snowdb.config import DATASET_CONFIG_FILENAME
    from snowtool.snowdb.datasets import config_from_spec

    src = tmp_path / 'src'
    _write_aoi(src, '33333:MT:USGS', polygon=_STRADDLE)
    _write_aoi(src, '55555:MT:USGS', polygon=_OUTSIDE)
    manager.pourpoints.import_(src)

    other = make_spec('other', spec)
    config = config_from_spec(other)
    ds_dir = manager.db.dataset_dir('other', config)
    ds_dir.mkdir(parents=True, exist_ok=True)
    config_path = ds_dir / DATASET_CONFIG_FILENAME
    config.save(config_path)

    staged = manager.stage_dataset('other', config_path)

    # Coverage classifies both basins; only the (partially) served one burns.
    # The wholly off-grid basin is still handed to rasterize_aois, whose own
    # Coverage.NONE check skips it -- so it appears in `skipped`, not omitted.
    assert staged.coverage == {
        '33333:MT:USGS': Coverage.PARTIAL,
        '55555:MT:USGS': Coverage.NONE,
    }
    assert staged.rasterized.built == [('33333:MT:USGS', 'other')]
    assert staged.rasterized.skipped == [('55555:MT:USGS', 'other')]
    aoi_dir = staged.dataset.path / 'aoi-rasters'
    assert (aoi_dir / '33333_MT_USGS.tif').is_file()
    assert not (aoi_dir / '55555_MT_USGS.tif').exists()
