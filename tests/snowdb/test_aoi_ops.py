"""SnowDb AOI import/sync/dump/remove + Dataset staleness/cascade helpers."""

import json

import pytest

from snowtool.exceptions import AOIPruneDestinationRequiredError
from snowtool.snowdb.aoi import AOI
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.db import SnowDb

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
def snowdb(tmp_path, spec, source_dem):
    """An initialized snowdb with the synthetic 'test' dataset created (has a DEM)."""
    SnowDb.initialize(tmp_path, [spec])
    Dataset.create(spec, tmp_path / 'data' / 'test', source_dem)
    return SnowDb(tmp_path, [spec])


# --- import ------------------------------------------------------------------


def test_import_file_writes_record_and_index(snowdb, aoi_geojson):
    result = snowdb.import_aois(aoi_geojson)

    assert result.imported == ['12345:MT:USGS']
    assert snowdb.aoi_record_path('12345:MT:USGS').is_file()
    assert snowdb.aoi_index().triplets() == {'12345:MT:USGS'}


def test_import_dir_classifies_imported_skipped_invalid(snowdb, tmp_path):
    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')
    _write_aoi(src, '22222:MT:USGS', with_polygon=False)  # point-only -> skipped
    (src / 'bad.geojson').write_text(json.dumps({'type': 'Nonsense'}))

    result = snowdb.import_aois(src)

    assert result.imported == ['11111:MT:USGS']
    assert result.skipped == ['22222:MT:USGS']
    assert [p.name for p, _ in result.invalid] == ['bad.geojson']
    assert snowdb._stored_triplets() == {'11111:MT:USGS'}


def test_import_dry_run_writes_nothing(snowdb, aoi_geojson):
    result = snowdb.import_aois(aoi_geojson, dry_run=True)

    assert result.imported == ['12345:MT:USGS']
    assert not snowdb.aoi_record_path('12345:MT:USGS').exists()
    assert not snowdb.aoi_index_path.exists()


def test_import_is_idempotent(snowdb, aoi_geojson):
    snowdb.import_aois(aoi_geojson)
    snowdb.import_aois(aoi_geojson)

    assert snowdb._stored_triplets() == {'12345:MT:USGS'}


# --- sync --------------------------------------------------------------------


def test_sync_prunes_absent_aoi_and_cascades(snowdb, tmp_path):
    # Two stored AOIs; sync a source dir that only has one.
    snowdb.import_aois(_write_aoi(tmp_path / 'seed', '11111:MT:USGS').parent)
    snowdb.import_aois(_write_aoi(tmp_path / 'seed', '22222:MT:USGS').parent)
    # Burn a raster for the one about to be pruned, to prove the cascade.
    snowdb.rasterize_aoi(snowdb.load_aoi('22222:MT:USGS'))
    raster = snowdb['test'].aoi_raster_path_from_triplet('22222:MT:USGS')
    assert raster.is_file()

    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')
    archive = tmp_path / 'archive'

    result = snowdb.sync_aois(src, prune_to=archive)

    assert result.pruned == ['22222:MT:USGS']
    assert not snowdb.aoi_record_path('22222:MT:USGS').exists()
    assert not raster.exists()  # cascade
    assert (archive / '22222_MT_USGS.geojson').is_file()  # dumped first
    assert snowdb.aoi_index().triplets() == {'11111:MT:USGS'}


def test_sync_without_prune_to_refuses_to_remove(snowdb, tmp_path):
    snowdb.import_aois(_write_aoi(tmp_path / 'seed', '22222:MT:USGS').parent)
    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')

    with pytest.raises(AOIPruneDestinationRequiredError):
        snowdb.sync_aois(src)

    # Nothing was changed (the additive import did not run either).
    assert snowdb._stored_triplets() == {'22222:MT:USGS'}
    assert not snowdb.aoi_record_path('11111:MT:USGS').exists()


def test_sync_dry_run_reports_prune_without_removing(snowdb, tmp_path):
    snowdb.import_aois(_write_aoi(tmp_path / 'seed', '22222:MT:USGS').parent)
    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')

    result = snowdb.sync_aois(src, dry_run=True)

    assert result.pruned == ['22222:MT:USGS']
    assert snowdb._stored_triplets() == {'22222:MT:USGS'}  # unchanged


# --- dump / remove / reindex -------------------------------------------------


def test_dump_aoi_copies_record_out(snowdb, aoi_geojson, tmp_path):
    snowdb.import_aois(aoi_geojson)
    dest = snowdb.dump_aoi('12345:MT:USGS', tmp_path / 'out')
    assert dest == tmp_path / 'out' / '12345_MT_USGS.geojson'
    assert dest.is_file()


def test_remove_aoi_cascades_and_reindexes(snowdb, aoi_geojson):
    snowdb.import_aois(aoi_geojson)
    snowdb.rasterize_aoi(snowdb.load_aoi('12345:MT:USGS'))
    raster = snowdb['test'].aoi_raster_path_from_triplet('12345:MT:USGS')
    assert raster.is_file()

    assert snowdb.remove_aoi('12345:MT:USGS') is True
    assert not snowdb.aoi_record_path('12345:MT:USGS').exists()
    assert not raster.exists()
    assert snowdb.aoi_index().triplets() == set()


def test_remove_absent_aoi_is_a_noop(snowdb):
    assert snowdb.remove_aoi('99999:MT:USGS') is False


def test_reindex_rebuilds_from_records(snowdb, aoi_geojson):
    snowdb.import_aois(aoi_geojson)
    snowdb.aoi_index_path.unlink()

    index = snowdb.reindex_aois()
    assert index.triplets() == {'12345:MT:USGS'}
    assert snowdb.aoi_index_path.is_file()


# --- Dataset staleness + cascade primitives ----------------------------------


def test_aoi_raster_hash_matches_aoi(snowdb, aoi_geojson):
    aoi = AOI.from_geojson(aoi_geojson)
    snowdb['test'].rasterize_aoi(aoi)
    assert snowdb['test'].aoi_raster_hash('12345:MT:USGS') == aoi.geometry_hash
    assert snowdb['test'].aoi_raster_is_current(aoi)


def test_aoi_raster_hash_none_when_absent(snowdb):
    assert snowdb['test'].aoi_raster_hash('99999:MT:USGS') is None


def test_rasterize_if_needed_builds_then_skips_then_rebuilds(
    snowdb,
    aoi_geojson,
    tmp_path,
):
    ds = snowdb['test']
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


def test_remove_aoi_raster(snowdb, aoi_geojson):
    ds = snowdb['test']
    ds.rasterize_aoi(AOI.from_geojson(aoi_geojson))
    assert ds.remove_aoi_raster('12345:MT:USGS') is True
    assert ds.remove_aoi_raster('12345:MT:USGS') is False  # idempotent


def test_rasterize_aois_built_and_skipped(snowdb, aoi_geojson):
    aoi = AOI.from_geojson(aoi_geojson)
    datasets = list(snowdb.datasets.values())

    first = snowdb.rasterize_aois([aoi], datasets)
    assert first.built == [('12345:MT:USGS', 'test')]
    assert first.skipped == []

    second = snowdb.rasterize_aois([aoi], datasets)
    assert second.built == []
    assert second.skipped == [('12345:MT:USGS', 'test')]
