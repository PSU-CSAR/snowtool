"""AOI.geometry_hash and the derived AOIIndex (FeatureCollection manifest)."""

import json

import pytest

from snowtool.snowdb.aoi import AOI
from snowtool.snowdb.aoi_index import AOIIndex, AOIIndexEntry


def _box(x0=-119.9, y0=44.9, x1=-119.0, y1=44.0):
    """A rectangular Polygon geometry inside the synthetic grid's first tile."""
    return {
        'type': 'Polygon',
        'coordinates': [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
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
    point = {'type': 'Point', 'coordinates': [-119.45, 44.45]}
    polygon = _box()
    properties = {
        'name': name,
        'source': source,
        'active': active,
        'basinarea': basinarea,
    }
    if with_polygon:
        feature = {
            'type': 'GeometryCollection',
            'id': triplet,
            'geometries': [point, polygon],
            'properties': properties,
        }
    else:
        feature = {
            'type': 'Feature',
            'id': triplet,
            'geometry': point,
            'properties': properties,
        }
    path.write_text(json.dumps(feature))
    return path


# --- AOI.geometry_hash -------------------------------------------------------


def test_geometry_hash_is_stable_and_hex(aoi_geojson):
    aoi = AOI.from_geojson(aoi_geojson)
    digest = aoi.geometry_hash
    assert digest == AOI.from_geojson(aoi_geojson).geometry_hash
    assert len(digest) == 64
    int(digest, 16)  # hex


def test_geometry_hash_differs_for_a_different_polygon(tmp_path):
    a = AOI.from_geojson(_write_pourpoint(tmp_path / 'a.geojson'))
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
    assert a.geometry_hash != AOI.from_geojson(b_path).geometry_hash


def test_geometry_hash_raises_without_a_polygon(tmp_path):
    path = _write_pourpoint(tmp_path / 'p.geojson', with_polygon=False)
    aoi = AOI.from_geojson(path)
    with pytest.raises(ValueError, match='does not have a polygon'):
        _ = aoi.geometry_hash


# --- AOIIndexEntry -----------------------------------------------------------


def test_entry_from_aoi_denormalizes_list_fields(aoi_geojson):
    entry = AOIIndexEntry.from_aoi(AOI.from_geojson(aoi_geojson))
    assert entry.triplet == '12345:MT:USGS'
    assert entry.name == 'Test Basin'
    assert entry.source == 'test'
    assert entry.point['type'] == 'Point'
    assert len(entry.geometry_hash) == 64


def test_entry_feature_round_trips(tmp_path):
    aoi = AOI.from_geojson(_write_pourpoint(tmp_path / 'p.geojson'))
    entry = AOIIndexEntry.from_aoi(aoi)
    feature = entry.to_feature()
    assert feature['type'] == 'Feature'
    assert feature['id'] == '12345:MT:USGS'
    assert feature['geometry']['type'] == 'Point'
    assert feature['properties']['active'] is True
    assert feature['properties']['basinarea'] == 5.2
    assert AOIIndexEntry.from_feature(feature) == entry


# --- AOIIndex ----------------------------------------------------------------


def test_index_from_records_and_save_load_round_trip(tmp_path):
    records = tmp_path / 'records'
    records.mkdir()
    _write_pourpoint(records / 'b.geojson', triplet='20000:MT:USGS')
    _write_pourpoint(records / 'a.geojson', triplet='10000:MT:USGS')

    index = AOIIndex.from_records(records)
    assert index.triplets() == {'10000:MT:USGS', '20000:MT:USGS'}

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

    reloaded = AOIIndex.load(out)
    assert reloaded.entries == index.entries


def test_index_from_records_skips_point_only_records(tmp_path):
    # A point-only pourpoint (no basin) in records/ must not crash reindex on
    # geometry_hash; it is simply not an AOI and is left out of the index.
    records = tmp_path / 'records'
    records.mkdir()
    _write_pourpoint(records / 'a.geojson', triplet='10000:MT:USGS')
    _write_pourpoint(
        records / 'p.geojson',
        triplet='20000:MT:USGS',
        with_polygon=False,
    )

    index = AOIIndex.from_records(records)
    assert index.triplets() == {'10000:MT:USGS'}


def test_index_from_records_empty_when_dir_absent(tmp_path):
    assert AOIIndex.from_records(tmp_path / 'nope').triplets() == set()


def test_index_load_missing_file_is_empty(tmp_path):
    assert len(AOIIndex.load(tmp_path / 'index.geojson')) == 0
