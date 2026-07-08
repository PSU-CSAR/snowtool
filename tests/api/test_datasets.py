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
    datasets = response.json()['datasets']
    assert [d['name'] for d in datasets] == ['instarr', 'snodas', 'swann-800m']
    # Each item is a followable resource: it carries its own self link to the
    # detail route, not a link stranded in the collection-level links array.
    for item in datasets:
        self_link = next(link for link in item['links'] if link['rel'] == 'self')
        assert self_link['href'].endswith(f'/datasets/{item["name"]}')


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


def test_get_dataset_info_advertises_zones(snodas_client) -> None:
    body = snodas_client.get('/datasets/snodas').json()
    zones = {z['key']: z for z in body['zones']}

    # Every stratifiable layer, sorted by key, with its scheme kind + override param.
    assert set(zones) == {
        'landcover.forest_cover',
        'terrain.aspect',
        'terrain.aspect_entropy',
        'terrain.eastness',
        'terrain.elevation',
        'terrain.northness',
    }
    assert body['zones'] == sorted(body['zones'], key=lambda z: z['key'])

    # Banded elevation: overridable by band_step_ft (int), default 1000 ft.
    elevation = zones['terrain.elevation']
    assert (
        elevation['kind'],
        elevation['param'],
        elevation['default'],
        elevation['unit'],
    ) == (
        'banded',
        'band_step_ft',
        1000,
        'ft',
    )
    assert elevation['classes'] is None

    # Aspect-orientation components: banded, overridable by band_step_pct.
    assert zones['terrain.northness']['param'] == 'band_step_pct'
    assert zones['terrain.eastness']['param'] == 'band_step_pct'

    # Threshold layers: forest cover + aspect entropy carry their own params.
    assert zones['landcover.forest_cover']['kind'] == 'threshold'
    assert zones['landcover.forest_cover']['param'] == 'threshold_pct'
    assert zones['terrain.aspect_entropy']['param'] == 'entropy_threshold'

    # Categorical aspect: no override param, but advertises its class keys/labels.
    aspect = zones['terrain.aspect']
    assert aspect['kind'] == 'categorical'
    assert aspect['param'] is None
    assert aspect['default'] is None
    assert [c['key'] for c in aspect['classes']] == ['N', 'E', 'S', 'W', 'flat']


def test_get_unknown_dataset_returns_404(test_client) -> None:
    response = test_client.get('/datasets/nope')
    assert response.status_code == 404
    # The problem carries the registered, resolvable type URI -- not about:blank.
    assert response.json()['type'] == '/problems/dataset-not-found'
