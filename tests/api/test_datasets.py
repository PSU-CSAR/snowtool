from collections.abc import Iterator

import pytest

from fastapi.testclient import TestClient

from snowtool.api.app import get_app


@pytest.fixture
def snodas_client(test_settings) -> Iterator[TestClient]:
    """A client whose snow database has a (content-free) snodas dataset dir.

    DatasetInfo is derived entirely from the spec, so an empty directory is
    enough for SnowDb to discover and bind the snodas dataset.
    """
    (test_settings.snowdb_path / 'data' / 'snodas').mkdir(parents=True)
    with TestClient(get_app(settings=test_settings)) as client:
        yield client


def test_list_datasets_empty(test_client) -> None:
    response = test_client.get('/datasets')
    assert response.status_code == 200
    assert response.json()['datasets'] == []


def test_list_datasets_with_snodas(snodas_client) -> None:
    response = snodas_client.get('/datasets')
    assert response.status_code == 200
    assert response.json()['datasets'] == ['snodas']


def test_get_dataset_info(snodas_client) -> None:
    response = snodas_client.get('/datasets/snodas')
    assert response.status_code == 200
    body = response.json()

    assert body['name'] == 'snodas'
    assert body['grid']['crs'] == '4326'
    assert body['grid']['is_geographic'] is True
    assert body['grid']['rows'] == 3351
    assert body['grid']['cols'] == 6935

    variables = body['variables']
    assert len(variables) == 8
    assert 'swe' in {v['key'] for v in variables}
    assert 'mean_swe_mm' in {v['stat_name'] for v in variables}


def test_get_unknown_dataset_returns_404(test_client) -> None:
    response = test_client.get('/datasets/nope')
    assert response.status_code == 404
