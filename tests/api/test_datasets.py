from collections.abc import Iterator

import pytest

from fastapi.testclient import TestClient

from snowtool.api.app import get_app

from ..conftest import init_with_builtins


@pytest.fixture
def snodas_client(test_settings) -> Iterator[TestClient]:
    """A client whose snow database has the built-in datasets registered.

    DatasetInfo is derived entirely from the spec, so no COGs are needed; the
    snodas dataset is served because its config is registered in the root config.
    """

    init_with_builtins(test_settings.snowdb_config)
    with TestClient(get_app(settings=test_settings)) as client:
        yield client


def test_list_datasets_lists_registered(test_client) -> None:
    # The app serves exactly the datasets registered in the root config (the
    # built-ins, here), sorted.
    response = test_client.get('/datasets')
    assert response.status_code == 200
    assert response.json()['datasets'] == ['instarr', 'snodas', 'swann-800m']


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
