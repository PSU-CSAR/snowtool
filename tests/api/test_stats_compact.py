"""API tests for the generic compact zonal-stats endpoint over a synthetic snowdb.

The synthetic ``test`` dataset is uniform (SWE 50, elevation 1000 m -> 3000-4000
ft, forest 100%), so the compact matrix is hand-computable. One generic route with
{dataset} a path param; zone selection uses LAYER:PARAM=VALUE tokens.
"""

import pytest

from gazebo.testing import assert_problem

from ..conftest import SWE_VALUE

TRIPLET = '12345:MT:USGS'
BASE = f'/datasets/test/stats-compact/{TRIPLET}'
DAY = '2018-04-27/2018-04-27'


def test_compact_whole_basin(synthetic_client) -> None:
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


def test_compact_elevation_override_token(synthetic_client) -> None:
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


def test_compact_include_empty_zones(synthetic_client) -> None:
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


def test_compact_doy(synthetic_client) -> None:
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


def test_compact_unknown_dataset_404(synthetic_client) -> None:
    response = synthetic_client.get(
        f'/datasets/nope/stats-compact/{TRIPLET}/date-range',
        params={'datetime': DAY},
    )
    assert_problem(response, status=404)


def test_compact_bad_zone_token_422(synthetic_client) -> None:
    response = synthetic_client.get(
        f'{BASE}/date-range',
        params={'datetime': DAY, 'zone': 'terrain.elevation:500'},
    )
    assert_problem(response, status=422)


def test_compact_unknown_variable_422(synthetic_client) -> None:
    response = synthetic_client.get(
        f'{BASE}/date-range',
        params={'datetime': DAY, 'variable': 'nope'},
    )
    assert_problem(response, status=422)
