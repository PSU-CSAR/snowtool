"""API tests for the canonical generic zonal-stats endpoint over a synthetic snowdb.

The synthetic ``test`` dataset is uniform (SWE 50, elevation 1000 m -> the
3000-4000 ft band, forest 100%), so whole-basin and crossed-zone results are
hand-computable -- mirroring the reader/CLI stats tests one HTTP layer up. One
generic route with ``{dataset}`` a path param; zone selection uses
``LAYER:PARAM=VALUE`` tokens. JSON (the compact body) vs csv is content-negotiated
(``?f=`` / ``Accept``).
"""

import pytest

from fastapi.testclient import TestClient
from gazebo.testing import assert_has_link, assert_problem

from snowtool.api.app import get_app

from ..conftest import SWE_VALUE, populate_synthetic_root

TRIPLET = '12345:MT:USGS'
BASE = f'/datasets/test/stats/{TRIPLET}'
DAY = '2018-04-27/2018-04-27'


def test_date_range_whole_basin_compact(synthetic_client) -> None:
    response = synthetic_client.get(
        f'{BASE}/date-range',
        params={'datetime': DAY, 'variable': 'swe'},
    )
    assert response.status_code == 200
    body = response.json()
    assert body['pourpoint'] == TRIPLET
    assert body['dataset'] == 'test'
    assert body['zone_layers'] == []
    assert body['variables'] == ['mean_swe_mm']
    (zone,) = body['zones']
    assert zone['zone'] == []
    assert zone['area_m2'] > 0
    (matrix,) = body['results'].values()
    assert matrix == [[pytest.approx(SWE_VALUE)]]
    # The JSON view advertises the CSV alternate.
    assert_has_link(body, 'alternate', type='text/csv')


def test_elevation_override_token(synthetic_client) -> None:
    response = synthetic_client.get(
        f'{BASE}/date-range',
        params={
            'datetime': DAY,
            'variable': 'swe',
            'zone': 'terrain.elevation:band_step_ft=2000',
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body['zone_layers'] == ['terrain.elevation']
    (zone,) = [z for z in body['zones'] if z['area_m2'] > 0]
    (ref,) = zone['zone']
    assert (ref['min'], ref['max']) == (2000, 4000)


def test_include_empty_zones(synthetic_client) -> None:
    response = synthetic_client.get(
        f'{BASE}/date-range',
        params={
            'datetime': DAY,
            'variable': 'swe',
            'zone': 'terrain.elevation',
            'include_empty_zones': 'true',
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body['zones']) == 16
    (matrix,) = body['results'].values()
    assert sum(1 for row in matrix if row == [None]) == 15


def test_day_of_year_compact(synthetic_client) -> None:
    response = synthetic_client.get(
        f'{BASE}/doy',
        params={
            'month': 4,
            'day': 27,
            'start_year': 2018,
            'end_year': 2018,
            'variable': 'swe',
        },
    )
    assert response.status_code == 200
    (matrix,) = response.json()['results'].values()
    assert matrix == [[pytest.approx(SWE_VALUE)]]


def test_csv_via_format_key(synthetic_client) -> None:
    response = synthetic_client.get(
        f'{BASE}/date-range',
        params={
            'datetime': DAY,
            'variable': 'swe',
            'zone': 'terrain.elevation',
            'f': 'csv',
        },
    )
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/csv')
    assert 'attachment; filename=' in response.headers['content-disposition']
    lines = response.text.strip().splitlines()
    assert lines[0] == (
        'date,terrain.elevation_min_ft,terrain.elevation_max_ft,area_m2,mean_swe_mm'
    )
    # Empty bands dropped by default -> the single populated 3000-4000 ft row.
    rows = lines[1:]
    (row,) = rows
    assert row.startswith('2018-04-27,3000,4000,')


def test_csv_via_accept_header(synthetic_client) -> None:
    response = synthetic_client.get(
        f'{BASE}/date-range',
        params={'datetime': DAY, 'variable': 'swe'},
        headers={'accept': 'text/csv'},
    )
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/csv')


def test_absent_datetime_returns_all_dates(synthetic_client) -> None:
    # No datetime filter -> every ingested date (the synthetic snowdb has one).
    response = synthetic_client.get(f'{BASE}/date-range', params={'variable': 'swe'})
    assert response.status_code == 200
    results = response.json()['results']
    assert len(results) == 1
    (matrix,) = results.values()
    assert matrix == [[pytest.approx(SWE_VALUE)]]


def test_open_ended_interval_selects_available(synthetic_client) -> None:
    # An open end (start/..) filters the available dates from start onward.
    response = synthetic_client.get(
        f'{BASE}/date-range',
        params={'datetime': '2018-01-01/..', 'variable': 'swe'},
    )
    assert response.status_code == 200
    (matrix,) = response.json()['results'].values()
    assert matrix == [[pytest.approx(SWE_VALUE)]]


def test_interval_outside_data_returns_empty(synthetic_client) -> None:
    # A range that selects no ingested date -> empty results, not an error.
    response = synthetic_client.get(
        f'{BASE}/date-range',
        params={'datetime': '2099-01-01/2099-12-31', 'variable': 'swe'},
    )
    assert response.status_code == 200
    assert response.json()['results'] == {}


def test_malformed_datetime_returns_400(synthetic_client) -> None:
    response = synthetic_client.get(
        f'{BASE}/date-range',
        params={'datetime': 'not-a-date', 'variable': 'swe'},
    )
    assert_problem(response, status=400)


def test_unknown_format_returns_400(synthetic_client) -> None:
    response = synthetic_client.get(
        f'{BASE}/date-range',
        params={'datetime': DAY, 'f': 'xml'},
    )
    assert_problem(response, status=400)


def test_bad_zone_token_returns_422(synthetic_client) -> None:
    # ``zone`` tokens are parsed by the handler against the dataset's registry, so a
    # malformed token is a well-formed-but-unprocessable query -- a 422.
    response = synthetic_client.get(
        f'{BASE}/date-range',
        params={'datetime': DAY, 'zone': 'terrain.elevation:500'},
    )
    assert_problem(response, status=422)


def test_unknown_variable_returns_422(synthetic_client) -> None:
    response = synthetic_client.get(
        f'{BASE}/date-range',
        params={'datetime': DAY, 'variable': 'nope'},
    )
    assert_problem(response, status=422)


def test_unknown_dataset_returns_404(synthetic_client) -> None:
    response = synthetic_client.get(
        f'/datasets/nope/stats/{TRIPLET}/date-range',
        params={'datetime': DAY},
    )
    assert_problem(response, status=404)


def test_unknown_aoi_returns_404(synthetic_client) -> None:
    # No stored record for this triplet -> the coverage guard's load_pourpoint 404s.
    response = synthetic_client.get(
        '/datasets/test/stats/00000:MT:USGS/date-range',
        params={'datetime': DAY, 'variable': 'swe'},
    )
    assert_problem(response, status=404)


def test_doy_impossible_day_returns_422(synthetic_client) -> None:
    # Feb 30 can occur in no year: DOYQuery's validator raises, and the route must
    # translate that into a 422 problem rather than let it 500.
    response = synthetic_client.get(
        f'{BASE}/doy',
        params={'month': 2, 'day': 30, 'start_year': 2018, 'end_year': 2018},
    )
    body = assert_problem(response, status=422)
    assert 'Invalid day of year' in body['detail']


def test_doy_inverted_year_span_returns_422(synthetic_client) -> None:
    response = synthetic_client.get(
        f'{BASE}/doy',
        params={'month': 4, 'day': 27, 'start_year': 2019, 'end_year': 2018},
    )
    body = assert_problem(response, status=422)
    assert 'Invalid day of year' in body['detail']


def test_uncovered_aoi_returns_409(
    test_settings,
    spec,
    pourpoint_geojson,
    tmp_path,
) -> None:
    # An AOI whose basin sits entirely off the dataset grid -> coverage 409.
    import json

    from snowtool.snowdb.manager import SnowDbManager

    populate_synthetic_root(test_settings.snowdb_config, spec, pourpoint_geojson)

    off_grid_triplet = '99999:MT:USGS'
    off_grid = {
        'type': 'GeometryCollection',
        'id': off_grid_triplet,
        'geometries': [
            {'type': 'Point', 'coordinates': [-100.5, 30.5]},
            {
                'type': 'Polygon',
                'coordinates': [
                    [
                        [-100.9, 30.9],
                        [-100.0, 30.9],
                        [-100.0, 30.0],
                        [-100.9, 30.0],
                        [-100.9, 30.9],
                    ],
                ],
            },
        ],
        'properties': {'name': 'Off Grid', 'source': 'test'},
    }
    path = tmp_path / 'off_grid.geojson'
    path.write_text(json.dumps(off_grid))
    SnowDbManager.open(test_settings.snowdb_config).import_pourpoints(path)

    with TestClient(get_app(settings=test_settings)) as client:
        response = client.get(
            f'/datasets/test/stats/{off_grid_triplet}/date-range',
            params={'datetime': DAY, 'variable': 'swe'},
        )
    assert_problem(response, status=409)


def test_incomplete_data_returns_500(test_settings, spec, pourpoint_geojson) -> None:
    # A stale duplicate swe COG (a leftover from a differently-named old source)
    # makes the swe variable unresolvable for the date -> a server data-integrity
    # failure surfaced as an informative 500 problem, not a bare 500.
    from datetime import date

    import numpy

    from snowtool.snowdb.raster.cog import write_cog

    from ..conftest import SIZE, TILE

    db = populate_synthetic_root(test_settings.snowdb_config, spec, pourpoint_geojson)
    dataset = db.datasets['test']
    date_dir = dataset.date_dir(date(2018, 4, 27))
    write_cog(
        date_dir / 'us_ssmv01034XdupTTNATS20180427.tif',
        numpy.full((SIZE, SIZE), SWE_VALUE, dtype=numpy.int16),
        transform=dataset.grid.base_grid.transform,
        tile_size=TILE,
        predictor=2,
    )

    with TestClient(get_app(settings=test_settings)) as client:
        response = client.get(
            f'{BASE}/date-range',
            params={'datetime': DAY, 'variable': 'swe'},
        )

    body = assert_problem(response, status=500)
    assert body['type'].endswith('/problems/incomplete-dataset-data')
    assert 'swe' in body['detail']
