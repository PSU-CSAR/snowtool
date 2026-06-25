"""SnowDb AOI import/sync/dump/remove + Dataset staleness/cascade helpers."""

import json

import pytest

import snowtool.snowdb.dataset as dataset_mod

from snowtool.exceptions import AOIPruneDestinationRequiredError
from snowtool.snowdb.aoi import AOI
from snowtool.snowdb.dataset import Dataset, aoi_provenance
from snowtool.snowdb.manager import SnowDbManager

from ..conftest import make_manager

_POINT = {'type': 'Point', 'coordinates': [-119.45, 44.45]}


def _box(x0=-119.9, y0=44.9, x1=-119.0, y1=44.0):
    """A rectangular Polygon geometry inside the synthetic grid's first tile."""
    return {
        'type': 'Polygon',
        'coordinates': [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }


# A polygon well inside the synthetic grid's first tile (see top-level conftest).
_POLYGON = _box()


def _write_aoi(directory, triplet, *, with_polygon=True, polygon=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{triplet.replace(":", "_")}.geojson'
    properties = {'name': triplet, 'source': 'test', 'active': True, 'basinarea': 5.2}
    if with_polygon:
        feature = {
            'type': 'GeometryCollection',
            'id': triplet,
            'geometries': [_POINT, polygon or _POLYGON],
            'properties': properties,
        }
    else:
        feature = {
            'type': 'Feature',
            'id': triplet,
            'geometry': _POINT,
            'properties': properties,
        }
    path.write_text(json.dumps(feature))
    return path


@pytest.fixture
def manager(tmp_path, spec, source_dem):
    """An initialized snowdb (write side) with the synthetic 'test' dataset.

    Writes go through this manager; reads use the derived ``db`` fixture.
    """
    SnowDbManager.initialize(tmp_path, [spec])
    Dataset.create(spec, tmp_path / 'data' / 'test', source_dem)
    return make_manager(tmp_path, [spec])


@pytest.fixture
def db(manager):
    """The manager's read SnowDb (same underlying root)."""
    return manager.db


# --- import ------------------------------------------------------------------


def test_import_file_writes_record_and_index(manager, db, aoi_geojson):
    result = manager.import_aois(aoi_geojson)

    assert result.imported == ['12345:MT:USGS']
    assert db.aoi_record_path('12345:MT:USGS').is_file()
    assert db.aoi_index().triplets() == {'12345:MT:USGS'}


def test_import_dir_classifies_imported_skipped_invalid(manager, tmp_path):
    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')
    _write_aoi(src, '22222:MT:USGS', with_polygon=False)  # point-only -> skipped
    (src / 'bad.geojson').write_text(json.dumps({'type': 'Nonsense'}))

    result = manager.import_aois(src)

    assert result.imported == ['11111:MT:USGS']
    assert result.skipped == ['22222:MT:USGS']
    assert [p.name for p, _ in result.invalid] == ['bad.geojson']
    assert manager._stored_triplets() == {'11111:MT:USGS'}


def test_import_dry_run_writes_nothing(manager, db, aoi_geojson):
    result = manager.import_aois(aoi_geojson, dry_run=True)

    assert result.imported == ['12345:MT:USGS']
    assert not db.aoi_record_path('12345:MT:USGS').exists()
    assert not db.aoi_index_path.exists()


def test_import_is_idempotent(manager, aoi_geojson):
    manager.import_aois(aoi_geojson)
    manager.import_aois(aoi_geojson)

    assert manager._stored_triplets() == {'12345:MT:USGS'}


# --- sync --------------------------------------------------------------------


def test_sync_prunes_absent_aoi_and_cascades(manager, db, tmp_path):
    # Two stored AOIs; sync a source dir that only has one.
    manager.import_aois(_write_aoi(tmp_path / 'seed', '11111:MT:USGS').parent)
    manager.import_aois(_write_aoi(tmp_path / 'seed', '22222:MT:USGS').parent)
    # Burn a raster for the one about to be pruned, to prove the cascade.
    manager.rasterize_aoi(db.load_aoi('22222:MT:USGS'))
    raster = db['test'].aoi_raster_path_from_triplet('22222:MT:USGS')
    assert raster.is_file()

    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')
    archive = tmp_path / 'archive'

    result = manager.sync_aois(src, prune_to=archive)

    assert result.pruned == ['22222:MT:USGS']
    assert not db.aoi_record_path('22222:MT:USGS').exists()
    assert not raster.exists()  # cascade
    assert (archive / '22222_MT_USGS.geojson').is_file()  # dumped first
    assert db.aoi_index().triplets() == {'11111:MT:USGS'}


def test_sync_without_prune_to_refuses_to_remove(manager, db, tmp_path):
    manager.import_aois(_write_aoi(tmp_path / 'seed', '22222:MT:USGS').parent)
    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')

    with pytest.raises(AOIPruneDestinationRequiredError):
        manager.sync_aois(src)

    # Nothing was changed (the additive import did not run either).
    assert manager._stored_triplets() == {'22222:MT:USGS'}
    assert not db.aoi_record_path('11111:MT:USGS').exists()


def test_sync_dry_run_reports_prune_without_removing(manager, tmp_path):
    manager.import_aois(_write_aoi(tmp_path / 'seed', '22222:MT:USGS').parent)
    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')

    result = manager.sync_aois(src, dry_run=True)

    assert result.pruned == ['22222:MT:USGS']
    assert manager._stored_triplets() == {'22222:MT:USGS'}  # unchanged


# --- dump / remove / reindex -------------------------------------------------


def test_dump_aoi_copies_record_out(manager, db, aoi_geojson, tmp_path):
    manager.import_aois(aoi_geojson)
    dest = db.dump_aoi('12345:MT:USGS', tmp_path / 'out')
    assert dest == tmp_path / 'out' / '12345_MT_USGS.geojson'
    assert dest.is_file()


def test_remove_aoi_cascades_and_reindexes(manager, db, aoi_geojson):
    manager.import_aois(aoi_geojson)
    manager.rasterize_aoi(db.load_aoi('12345:MT:USGS'))
    raster = db['test'].aoi_raster_path_from_triplet('12345:MT:USGS')
    assert raster.is_file()

    assert manager.remove_aoi('12345:MT:USGS') is True
    assert not db.aoi_record_path('12345:MT:USGS').exists()
    assert not raster.exists()
    assert db.aoi_index().triplets() == set()


def test_remove_absent_aoi_is_a_noop(manager):
    assert manager.remove_aoi('99999:MT:USGS') is False


def test_reindex_rebuilds_from_records(manager, db, aoi_geojson):
    manager.import_aois(aoi_geojson)
    db.aoi_index_path.unlink()

    index = manager.reindex_aois()
    assert index.triplets() == {'12345:MT:USGS'}
    assert db.aoi_index_path.is_file()


# --- Dataset staleness + cascade primitives ----------------------------------


def test_aoi_raster_hash_matches_aoi(db, aoi_geojson):

    aoi = AOI.from_geojson(aoi_geojson)
    db['test'].rasterize_aoi(aoi)
    # The stored tag is the versioned provenance (geometry hash + format version),
    # not the bare geometry hash.
    assert db['test'].aoi_raster_hash('12345:MT:USGS') == aoi_provenance(
        aoi.geometry_hash,
    )
    assert db['test'].aoi_raster_is_current(aoi)


def test_aoi_raster_hash_none_when_absent(db):
    assert db['test'].aoi_raster_hash('99999:MT:USGS') is None


def test_format_version_bump_makes_aoi_raster_stale(db, aoi_geojson, monkeypatch):
    # A material format change is detected by the same staleness check as a changed
    # basin: bump the burned-raster format version and an already-current raster
    # (unchanged geometry) reads as stale, forcing a rebuild.

    aoi = AOI.from_geojson(aoi_geojson)
    ds = db['test']
    ds.rasterize_aoi(aoi)
    assert ds.aoi_raster_is_current(aoi) is True

    monkeypatch.setattr(
        dataset_mod,
        'AOI_RASTER_FORMAT_VERSION',
        dataset_mod.AOI_RASTER_FORMAT_VERSION + 1,
    )
    assert ds.aoi_raster_is_current(aoi) is False
    assert ds.rasterize_aoi_if_needed(aoi) is True
    assert ds.aoi_raster_is_current(aoi) is True


def test_rasterize_if_needed_builds_then_skips_then_rebuilds(
    db,
    aoi_geojson,
    tmp_path,
):
    ds = db['test']
    aoi = AOI.from_geojson(aoi_geojson)

    assert ds.rasterize_aoi_if_needed(aoi) is True  # missing -> built
    assert ds.rasterize_aoi_if_needed(aoi) is False  # current -> skipped

    # A changed basin makes the existing raster stale.
    stale = _write_aoi(
        tmp_path / 'stale',
        '12345:MT:USGS',
        polygon=_box(-119.8, 44.8, -119.1, 44.1),
    )
    stale_aoi = AOI.from_geojson(stale)
    assert ds.aoi_raster_is_current(stale_aoi) is False
    assert ds.rasterize_aoi_if_needed(stale_aoi) is True  # stale -> rebuilt
    assert ds.rasterize_aoi_if_needed(stale_aoi, rebuild=False) is False


def test_remove_aoi_raster(db, aoi_geojson):
    ds = db['test']
    ds.rasterize_aoi(AOI.from_geojson(aoi_geojson))
    assert ds.remove_aoi_raster('12345:MT:USGS') is True
    assert ds.remove_aoi_raster('12345:MT:USGS') is False  # idempotent


def test_rasterize_aois_built_and_skipped(manager, db, aoi_geojson):
    aoi = AOI.from_geojson(aoi_geojson)
    datasets = list(db.datasets.values())

    first = manager.rasterize_aois([aoi], datasets)
    assert first.built == [('12345:MT:USGS', 'test')]
    assert first.skipped == []

    second = manager.rasterize_aois([aoi], datasets)
    assert second.built == []
    assert second.skipped == [('12345:MT:USGS', 'test')]
