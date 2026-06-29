"""API tests for the AOI listing + detail routes over a synthetic snowdb."""

import json

import pytest

from fastapi.testclient import TestClient
from gazebo.testing import (
    assert_has_link,
    assert_problem,
    drive_pagination,
    find_link,
)

from snowtool.api.app import get_app

TRIPLET = '12345:MT:USGS'


def _aoi_feature(triplet: str, west: float, south: float) -> dict:
    """A polygon-bearing pourpoint inside the synthetic grid (so it indexes)."""
    return {
        'type': 'GeometryCollection',
        'id': triplet,
        'geometries': [
            {'type': 'Point', 'coordinates': [west + 0.25, south + 0.25]},
            {
                'type': 'Polygon',
                'coordinates': [
                    [
                        [west, south + 0.5],
                        [west + 0.5, south + 0.5],
                        [west + 0.5, south],
                        [west, south],
                        [west, south + 0.5],
                    ],
                ],
            },
        ],
        'properties': {'name': triplet, 'source': 'test'},
    }


@pytest.fixture
def many_aois_client(test_settings, spec, tmp_path):
    """A client over a root with three indexed AOIs (no rasters/COGs needed)."""
    from snowtool.snowdb.datasets import config_from_spec
    from snowtool.snowdb.manager import SnowDbManager

    from ..conftest import register_dataset_config

    manager = SnowDbManager.initialize(test_settings.snowdb_config)
    register_dataset_config(manager, spec.name, config_from_spec(spec))
    manager = SnowDbManager.open(test_settings.snowdb_config)

    src = tmp_path / 'import_src'
    src.mkdir()
    for i in range(3):
        triplet = f'{1000 + i}:MT:USGS'
        feature = _aoi_feature(triplet, west=-119.0 + i * 0.5, south=44.0 - i * 0.5)
        (src / f'{1000 + i}_MT_USGS.geojson').write_text(json.dumps(feature))
    manager.import_aois(src)

    with TestClient(get_app(settings=test_settings)) as client:
        yield client


def test_list_aois_collection_shape(synthetic_client) -> None:
    response = synthetic_client.get('/aois')
    assert response.status_code == 200
    body = response.json()
    # OGC API Features collection envelope from gazebo's LinkedCollection.
    assert body['type'] == 'FeatureCollection'
    assert body['numberReturned'] == 1
    assert body['numberMatched'] == 1
    (feature,) = body['features']
    assert feature['type'] == 'Feature'
    assert feature['id'] == TRIPLET
    assert feature['geometry']['type'] == 'Point'
    # Per-dataset coverage is carried in the index; the basin is fully on-grid.
    assert feature['properties']['coverage'] == {'test': 'full'}
    # Each feature self-links to its detail route.
    (self_link,) = [link for link in feature['links'] if link['rel'] == 'self']
    assert self_link['href'].endswith(f'/aois/{TRIPLET}')


def test_get_aoi_detail(synthetic_client) -> None:
    response = synthetic_client.get(f'/aois/{TRIPLET}')
    assert response.status_code == 200
    body = response.json()
    assert body['id'] == TRIPLET
    # The full stored record: point + basin polygon as a GeometryCollection.
    assert body['geometry']['type'] == 'GeometryCollection'
    kinds = {geom['type'] for geom in body['geometry']['geometries']}
    assert kinds == {'Point', 'Polygon'}
    assert body['properties']['name'] == 'Test Basin'


def test_get_unknown_aoi_returns_404_problem(synthetic_client) -> None:
    assert_problem(synthetic_client.get('/aois/00000:MT:USGS'), status=404)


def test_get_aoi_invalid_triplet_returns_422(synthetic_client) -> None:
    # 'badtriplet' fails the StationTriplet path pattern -> request validation 422.
    assert_problem(synthetic_client.get('/aois/badtriplet'), status=422)


def test_list_aois_first_page_has_next_not_prev(many_aois_client) -> None:
    response = many_aois_client.get('/aois', params={'limit': 2})
    assert response.status_code == 200
    body = response.json()
    assert body['numberMatched'] == 3
    assert body['numberReturned'] == 2
    assert [f['id'] for f in body['features']] == ['1000:MT:USGS', '1001:MT:USGS']
    assert_has_link(body, 'self')
    assert_has_link(body, 'last')
    # A next page follows; no prev on the first page.
    assert 'offset=2' in assert_has_link(body, 'next')['href']
    assert find_link(body, 'prev') is None


def test_list_aois_last_page_has_prev_not_next(many_aois_client) -> None:
    response = many_aois_client.get('/aois', params={'limit': 2, 'offset': 2})
    assert response.status_code == 200
    body = response.json()
    assert body['numberReturned'] == 1
    assert [f['id'] for f in body['features']] == ['1002:MT:USGS']
    assert_has_link(body, 'prev')
    assert_has_link(body, 'first')
    assert find_link(body, 'next') is None


def test_pagination_walks_every_aoi(many_aois_client) -> None:
    # Drive the next links to exhaustion: every page's numberReturned must match its
    # item count, no page exceeds the limit, and the union is all three AOIs.
    features = drive_pagination(
        many_aois_client,
        '/aois?limit=2',
        items_key='features',
        limit=2,
    )
    assert [f['id'] for f in features] == [
        '1000:MT:USGS',
        '1001:MT:USGS',
        '1002:MT:USGS',
    ]


def test_list_aois_bbox_filters(many_aois_client) -> None:
    # The three AOIs span ~lon -119..-118; a bbox around only the first selects it.
    response = many_aois_client.get('/aois', params={'bbox': '-119.1,43.9,-118.6,44.6'})
    assert response.status_code == 200
    body = response.json()
    assert [f['id'] for f in body['features']] == ['1000:MT:USGS']
    assert body['numberMatched'] == 1


def test_list_aois_bad_bbox_returns_400(many_aois_client) -> None:
    assert_problem(many_aois_client.get('/aois', params={'bbox': '1,2,3'}), status=400)


def test_list_aois_limit_over_max_returns_422(many_aois_client) -> None:
    assert_problem(many_aois_client.get('/aois', params={'limit': 100000}), status=422)
