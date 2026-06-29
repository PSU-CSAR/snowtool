"""The OpenAPI document must build for the real multi-dataset surface.

The per-dataset stats routes are typed by each dataset's *generated*
``zonal_stats_model`` envelope. A bad generated schema would 500 ``/openapi.json``
(and ``/docs``) without touching the data routes, so the happy-path API tests
would miss it -- this asserts the document renders for the built-in datasets.
"""


def test_openapi_builds_for_builtin_datasets(test_client) -> None:
    response = test_client.get('/openapi.json')
    assert response.status_code == 200
    doc = response.json()

    # The per-dataset stats routes are present...
    paths = doc['paths']
    assert '/datasets/snodas/stats/{triplet}/date-range' in paths
    assert '/datasets/snodas/stats/{triplet}/doy' in paths

    # ...and each dataset's generated zonal-stats model surfaces as a real schema
    # (the model_prefix uniqueness check makes these names distinct per dataset).
    schemas = doc['components']['schemas']
    zonal_models = {name for name in schemas if 'ZonalStat' in name}
    assert any(name.startswith('Snodas') for name in zonal_models)
    # All three built-ins contribute a distinct prefix.
    prefixes = {name.split('ZonalStat')[0] for name in zonal_models}
    assert len(prefixes) >= 3
