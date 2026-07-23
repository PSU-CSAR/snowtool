"""Pourpoint.geometry_hash and the derived PourpointIndex (FeatureCollection)."""

import json

import pytest

from snowtool.exceptions import IndexedPourpointMissingBasinError
from snowtool.snowdb.coverage import Coverage, CoverageDomain
from snowtool.snowdb.grid import make_grid
from snowtool.snowdb.pourpoint import Pourpoint
from snowtool.snowdb.pourpoint_index import PourpointIndex, PourpointIndexEntry

from ..conftest import write_pourpoint_record


def _box(x0=-119.9, y0=44.9, x1=-119.0, y1=44.0):
    """A rectangular Polygon geometry inside the synthetic grid's first tile."""
    return {
        'type': 'Polygon',
        'coordinates': [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }


def _grids():
    """Two synthetic domains: one covering the test basin, one disjoint from it."""
    covers = make_grid(
        origin_x=-120.0,
        origin_y=45.0,
        px_size=0.01,
        cols=512,
        rows=512,
        tile_size=256,
        crs=4326,
    )
    disjoint = make_grid(
        origin_x=-100.0,
        origin_y=45.0,
        px_size=0.01,
        cols=512,
        rows=512,
        tile_size=256,
        crs=4326,
    )
    return {
        'covers': CoverageDomain.from_grid(covers, covers.crs),
        'disjoint': CoverageDomain.from_grid(disjoint, disjoint.crs),
    }


def _write_pourpoint(
    path,
    triplet='12345:MT:USGS',
    *,
    with_polygon=True,
    name='Test Basin',
    source='test',
    active=True,
    basinarea=5.2,
):
    return write_pourpoint_record(
        path,
        triplet,
        polygon=_box()['coordinates'][0],
        point_only=not with_polygon,
        properties={
            'name': name,
            'source': source,
            'active': active,
            'basinarea': basinarea,
        },
    )


# --- Pourpoint.geometry_hash -------------------------------------------------------


def test_geometry_hash_is_stable_and_hex(pourpoint_geojson):
    aoi = Pourpoint.from_geojson(pourpoint_geojson)
    digest = aoi.geometry_hash
    assert digest == Pourpoint.from_geojson(pourpoint_geojson).geometry_hash
    assert len(digest) == 64
    int(digest, 16)  # hex


def test_geometry_hash_differs_for_a_different_polygon(tmp_path):
    a = Pourpoint.from_geojson(_write_pourpoint(tmp_path / 'a.geojson'))
    shifted = _box(-118.9, 44.9, -118.0, 44.0)
    point = {'type': 'Point', 'coordinates': [-118.45, 44.45]}
    b_path = tmp_path / 'b.geojson'
    b_path.write_text(
        json.dumps(
            {
                'type': 'GeometryCollection',
                'id': '12345:MT:USGS',
                'geometries': [point, shifted],
                'properties': {'name': 'x', 'source': 'test'},
            },
        ),
    )
    assert a.geometry_hash != Pourpoint.from_geojson(b_path).geometry_hash


def test_geometry_hash_raises_without_a_polygon(tmp_path):
    path = _write_pourpoint(tmp_path / 'p.geojson', with_polygon=False)
    aoi = Pourpoint.from_geojson(path)
    # Twice: cached_property must not cache the raised error, only a result.
    for _ in range(2):
        with pytest.raises(ValueError, match='does not have a basin polygon'):
            _ = aoi.geometry_hash


def test_geometry_and_derived_values_are_cached(pourpoint_geojson):
    aoi = Pourpoint.from_geojson(pourpoint_geojson)
    # The shapely conversion runs once per instance: repeated access returns the
    # very same object (and so do the derived area/hash).
    assert aoi.geometry is aoi.geometry
    assert aoi.area_meters is aoi.area_meters
    assert aoi.geometry_hash is aoi.geometry_hash


def test_geometry_hash_pins_the_provenance_contract(pourpoint_geojson):
    # The exact sha256 of the fixture basin's canonical (little-endian) WKB.
    # This digest is baked into AOI-raster provenance tags; caching must never
    # change what is hashed or how.
    aoi = Pourpoint.from_geojson(pourpoint_geojson)
    assert aoi.geometry_hash == (
        'f8ae6d17bbe305616c4422b15d4116e988d766c6bc33bc4f4ee691f116ac7242'
    )


# --- PourpointIndexEntry -----------------------------------------------------------


def test_entry_from_aoi_denormalizes_list_fields(pourpoint_geojson):
    entry = PourpointIndexEntry.from_pourpoint(
        Pourpoint.from_geojson(pourpoint_geojson),
        _grids(),
    )
    assert entry.triplet == '12345:MT:USGS'
    assert entry.name == 'Test Basin'
    assert entry.point.type == 'Point'
    # Exact geodesic area (WGS84) of the fixture's 0.9 deg x 0.9 deg basin box.
    assert entry.area_meters == pytest.approx(7_164_269_879.72, rel=1e-9)
    assert len(entry.geometry_hash) == 64


def test_entry_from_aoi_computes_per_dataset_coverage(pourpoint_geojson):
    entry = PourpointIndexEntry.from_pourpoint(
        Pourpoint.from_geojson(pourpoint_geojson),
        _grids(),
    )
    assert entry.coverage == {
        'covers': Coverage.FULL,
        'disjoint': Coverage.NONE,
    }


# --- PourpointIndex ----------------------------------------------------------------


def test_index_build_and_save_load_round_trip(tmp_path):
    records = tmp_path / 'records'
    records.mkdir()
    _write_pourpoint(records / 'b.geojson', triplet='20000:MT:USGS')
    _write_pourpoint(records / 'a.geojson', triplet='10000:MT:USGS')

    index = PourpointIndex.build(sorted(records.glob('*.geojson')), _grids())
    assert index.triplets() == {'10000:MT:USGS', '20000:MT:USGS'}
    # Coverage is derived per dataset during the rebuild.
    assert index['10000:MT:USGS'].coverage == {
        'covers': Coverage.FULL,
        'disjoint': Coverage.NONE,
    }

    out = tmp_path / 'index.geojson'
    index.save(out)
    collection = json.loads(out.read_text())
    assert collection['type'] == 'FeatureCollection'
    # Features are sorted by triplet for stable diffs.
    assert [f['id'] for f in collection['features']] == [
        '10000:MT:USGS',
        '20000:MT:USGS',
    ]
    assert out.read_text().endswith('\n')

    reloaded = PourpointIndex.load(out)
    assert reloaded.entries == index.entries


def test_index_build_raises_on_point_only_record(tmp_path):
    # Every stored record is basin-bearing (the import boundary guarantees it), so
    # a point-only pourpoint in records/ is a corrupt store: build refuses loudly,
    # naming the offending file, rather than silently dropping the record.
    records = tmp_path / 'records'
    records.mkdir()
    _write_pourpoint(records / 'a.geojson', triplet='10000:MT:USGS')
    _write_pourpoint(
        records / 'p.geojson',
        triplet='20000:MT:USGS',
        with_polygon=False,
    )

    with pytest.raises(IndexedPourpointMissingBasinError, match=r'p\.geojson'):
        PourpointIndex.build(sorted(records.glob('*.geojson')), _grids())


def test_index_build_empty_over_no_paths(tmp_path):
    assert PourpointIndex.build([], _grids()).triplets() == set()


def test_index_load_missing_file_is_empty(tmp_path):
    assert len(PourpointIndex.load(tmp_path / 'index.geojson')) == 0
