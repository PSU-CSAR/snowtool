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


def test_get_dataset_info_advertises_templated_stats_links(snodas_client) -> None:
    # The dataset resource advertises templated links to its two stats query
    # endpoints, so a client can build a query from the dataset alone: the triplet is
    # an unbound path var and the query params (incl. each overridable zone's
    # ``<key>.<param>`` field) are an RFC 6570 form-query expansion.
    body = snodas_client.get('/datasets/snodas').json()
    links = {link['rel']: link for link in body['links']}

    date_range = links['stats-date-range']
    assert date_range['templated'] is True
    assert '/datasets/snodas/stats/{triplet}/date-range{?' in date_range['href']
    assert 'datetime' in date_range['href']
    # a per-dataset override param rides in the query template
    assert 'terrain.elevation.band_step_ft' in date_range['href']

    doy = links['stats-doy']
    assert doy['templated'] is True
    assert '/datasets/snodas/stats/{triplet}/doy{?' in doy['href']
    for var in ('month', 'day', 'start_year', 'end_year'):
        assert var in doy['href']


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
    assert 'classes' not in elevation
    # Its covered band range (the -100..4500 m bracket in feet, aligned to 1000 ft).
    assert (elevation['min'], elevation['max']) == (-1000, 15000)

    # Aspect-orientation components: bucketed (dimensionless [-1, 1]), overridable by
    # an integer bucket count, default 4, covering the [-1, 1] range; no unit field.
    for key in ('terrain.northness', 'terrain.eastness'):
        assert (
            zones[key]['kind'],
            zones[key]['param'],
            zones[key]['default'],
            zones[key]['min'],
            zones[key]['max'],
        ) == ('bucketed', 'buckets', 4, -1, 1)
        assert 'unit' not in zones[key]

    # Threshold layers carry their own params and advertise the range their split sits
    # within: forest cover 0..100 %, normalised aspect entropy 0..1.
    forest = zones['landcover.forest_cover']
    assert (
        forest['kind'],
        forest['param'],
        forest['min'],
        forest['max'],
    ) == ('threshold', 'threshold_pct', 0, 100)
    entropy = zones['terrain.aspect_entropy']
    assert (entropy['param'], entropy['min'], entropy['max']) == (
        'entropy_threshold',
        0,
        1,
    )

    # Categorical aspect: no override param, but advertises its class keys/labels.
    aspect = zones['terrain.aspect']
    assert aspect['kind'] == 'categorical'
    assert 'param' not in aspect
    assert 'default' not in aspect
    assert [c['key'] for c in aspect['classes']] == ['N', 'E', 'S', 'W', 'flat']


def test_get_unknown_dataset_returns_404(test_client) -> None:
    response = test_client.get('/datasets/nope')
    assert response.status_code == 404
    # The problem carries the registered, resolvable type URI -- not about:blank.
    assert response.json()['type'] == '/problems/dataset-not-found'
