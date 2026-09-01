"""API tests for the canonical generic zonal-stats endpoint over a synthetic snowdb.

The synthetic ``test`` dataset is uniform (SWE 50, elevation 1000 m -> the
3000-4000 ft band, forest 100%), so whole-basin and crossed-zone results are
hand-computable -- mirroring the reader/CLI stats tests one HTTP layer up. One
generic route with ``{dataset}`` a path param; zone selection uses
``LAYER:PARAM=VALUE`` tokens. JSON (the compact body) vs csv is content-negotiated
(``?f=`` / ``Accept``).
"""

import pytest

TRIPLET: str = '12345:MT:USGS'
GRADIENT_BASE: str = f'/datasets/test/snowline/{TRIPLET}'
DAY: str = '2018-04-27/2018-04-27'


def test_return_interpolated_snowline(gradient_client):
    response = gradient_client.get(
        f'{GRADIENT_BASE}/date-range',
        params={
            'datetime': DAY,
            'variable': 'swe',
            'threshold': 50,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body['pourpoint'] == TRIPLET
    assert body['dataset'] == 'test'

    (snowline,) = body['results'].values()
    assert snowline is not None
    assert snowline['elevation_ft'] == pytest.approx(4000.0, abs=1.0)
    assert snowline['lower_band_area_m2'] > 0
    assert snowline['higher_band_area_m2'] > 0
    assert snowline['interpolation_span_ft'] > 0


def test_return_query_params(gradient_client):
    response = gradient_client.get(
        f'{GRADIENT_BASE}/date-range',
        params={
            'datetime': DAY,
            'variable': 'swe',
            'threshold': 50,
            'band_step_ft': 1000,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body['threshold'] == 50
    assert body['threshold_unit'] == 'mm'
    assert body['band_step_ft'] == 1000


def test_different_snowline_thresholds(gradient_client):
    def get_elevation(threshold: float) -> float:
        response = gradient_client.get(
            f'{GRADIENT_BASE}/date-range',
            params={
                'datetime': DAY,
                'variable': 'swe',
                'threshold': threshold,
                'band_step_ft': 1000,
            },
        )
        print(response.json())
        assert response.status_code == 200
        body = response.json()
        (snowline,) = body['results'].values()
        return snowline['elevation_ft']

    snowline_a = get_elevation(70.5)
    snowline_b = get_elevation(33.7)

    assert snowline_a != snowline_b
    assert snowline_a > snowline_b


def test_return_none_above_threshold(gradient_client):
    response = gradient_client.get(
        f'{GRADIENT_BASE}/date-range',
        params={
            'datetime': DAY,
            'variable': 'swe',
            'threshold': 99999999,
        },
    )
    assert response.status_code == 200
    (snowline,) = response.json()['results'].values()
    assert snowline is None
