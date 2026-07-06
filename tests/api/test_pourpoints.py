"""API tests for the pourpoint listing + detail routes over a synthetic snowdb."""

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
    manager.import_pourpoints(src)

    with TestClient(get_app(settings=test_settings)) as client:
        yield client


@pytest.fixture
def inactive_dataset_client(test_settings, spec, pourpoint_geojson):
    """A client over a root with two registered datasets, one deactivated.

    The index carries coverage for both (the admin surface shows inactive
    coverage on purpose); the API responses must expose only the active key.
    """
    from snowtool.snowdb.datasets import config_from_spec
    from snowtool.snowdb.manager import SnowDbManager
    from snowtool.snowdb.spec import DatasetSpec

    from ..conftest import register_dataset_config

    root = test_settings.snowdb_config
    manager = SnowDbManager.initialize(root)
    register_dataset_config(manager, spec.name, config_from_spec(spec))
    other = DatasetSpec(
        name='other',
        grid_params=spec.grid_params,
        variables=spec.variables.values(),
    )
    register_dataset_config(manager, 'other', config_from_spec(other))
    manager = SnowDbManager.open(root)  # rebind so both datasets are served
    manager.import_pourpoints(pourpoint_geojson)
    manager.set_dataset_active('other', False)

    with TestClient(get_app(settings=test_settings)) as client:
        yield client


def test_coverage_is_filtered_to_active_datasets(
    inactive_dataset_client,
    test_settings,
) -> None:
    from snowtool.snowdb.db import SnowDb

    # List and detail responses carry coverage only for the active dataset: a
    # client must never see a key that /datasets and the stats routes 404 on.
    (feature,) = inactive_dataset_client.get('/pourpoints').json()['features']
    assert feature['properties']['coverage'] == {'test': 'full'}
    detail = inactive_dataset_client.get(f'/pourpoints/{TRIPLET}').json()
    assert detail['properties']['coverage'] == {'test': 'full'}

    # The index itself (the admin surface -- CLI `pourpoint list` serves it
    # verbatim) still carries the inactive dataset's coverage.
    index = SnowDb.open(test_settings.snowdb_config).pourpoint_index()
    assert set(index[TRIPLET].coverage) == {'test', 'other'}


def test_list_aois_collection_shape(synthetic_client) -> None:
    response = synthetic_client.get('/pourpoints')
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
    assert self_link['href'].endswith(f'/pourpoints/{TRIPLET}')


def test_get_aoi_detail(synthetic_client) -> None:
    response = synthetic_client.get(f'/pourpoints/{TRIPLET}')
    assert response.status_code == 200
    body = response.json()
    assert body['id'] == TRIPLET
    # Detail has no geometry param: it always returns the basin polygon.
    assert body['geometry']['type'] == 'Polygon'
    props = body['properties']
    assert props['name'] == 'Test Basin'
    # The outflow point rides along as a property even though the geometry is the
    # basin, and the geodesic area is computed from the polygon (exact, WGS84).
    assert len(props['pourpoint']) == 2
    assert props['area_meters'] == pytest.approx(7_164_269_879.72, rel=1e-9)
    # Coverage is pulled from the index by triplet.
    assert props['coverage'] == {'test': 'full'}
    # Curated ids are present (null here -- the synthetic record has none).
    assert props['awdb_id'] is None
    assert props['usgs_id'] is None


def test_list_basin_geometry_returns_polygons(synthetic_client) -> None:
    response = synthetic_client.get('/pourpoints', params={'geometry': 'basin'})
    assert response.status_code == 200
    (feature,) = response.json()['features']
    # Basin mode swaps the geometry slot to the polygon...
    assert feature['geometry']['type'] == 'Polygon'
    # ...but the pourpoint coordinate is still carried as a property.
    assert len(feature['properties']['pourpoint']) == 2


def test_list_point_geometry_default_carries_pourpoint(synthetic_client) -> None:
    (feature,) = synthetic_client.get('/pourpoints').json()['features']
    assert feature['geometry']['type'] == 'Point'
    # The point is both the geometry and the property in point mode.
    assert tuple(feature['properties']['pourpoint']) == tuple(
        feature['geometry']['coordinates'],
    )


def test_basin_geometry_survives_pagination(many_aois_client) -> None:
    # Basin mode + paging: the `next` link must carry geometry=basin, or page 2
    # silently falls back to point geometry. Assert the link *and* the followed
    # page so a regression in either the link-building or the param can't pass.
    page1 = many_aois_client.get(
        '/pourpoints',
        params={'geometry': 'basin', 'limit': 2},
    ).json()
    assert [f['geometry']['type'] for f in page1['features']] == ['Polygon', 'Polygon']
    next_href = assert_has_link(page1, 'next')['href']
    assert 'geometry=basin' in next_href

    page2 = many_aois_client.get(next_href).json()
    assert [f['geometry']['type'] for f in page2['features']] == ['Polygon']


def test_get_unknown_aoi_returns_404_problem(synthetic_client) -> None:
    assert_problem(synthetic_client.get('/pourpoints/00000:MT:USGS'), status=404)


def test_get_aoi_invalid_triplet_returns_422(synthetic_client) -> None:
    # 'badtriplet' fails the StationTriplet path pattern -> request validation 422.
    assert_problem(synthetic_client.get('/pourpoints/badtriplet'), status=422)


def test_list_aois_first_page_has_next_not_prev(many_aois_client) -> None:
    response = many_aois_client.get('/pourpoints', params={'limit': 2})
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
    response = many_aois_client.get('/pourpoints', params={'limit': 2, 'offset': 2})
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
        '/pourpoints?limit=2',
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
    response = many_aois_client.get(
        '/pourpoints',
        params={'bbox': '-119.1,43.9,-118.6,44.6'},
    )
    assert response.status_code == 200
    body = response.json()
    assert [f['id'] for f in body['features']] == ['1000:MT:USGS']
    assert body['numberMatched'] == 1


def test_list_aois_bad_bbox_returns_400(many_aois_client) -> None:
    assert_problem(
        many_aois_client.get('/pourpoints', params={'bbox': '1,2,3'}),
        status=400,
    )


def test_list_aois_limit_over_max_returns_422(many_aois_client) -> None:
    assert_problem(
        many_aois_client.get('/pourpoints', params={'limit': 100000}),
        status=422,
    )
