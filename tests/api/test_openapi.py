"""The OpenAPI document must build for the real multi-dataset surface.

The stats routes are one generic router (``{dataset}`` a path param) typed by the
single generic :class:`CompactStatsResponse` envelope. A bad schema would 500
``/openapi.json`` (and ``/docs``) without touching the data routes, so the
happy-path API tests would miss it -- this asserts the document renders for the
built-in datasets.
"""


def test_openapi_builds_for_builtin_datasets(test_client) -> None:
    response = test_client.get('/openapi.json')
    assert response.status_code == 200
    doc = response.json()

    # The one generic stats route family is present (not per-dataset).
    paths = doc['paths']
    assert '/datasets/{dataset}/stats/{triplet}/date-range' in paths
    assert '/datasets/{dataset}/stats/{triplet}/doy' in paths

    # A single generic response schema types every dataset's stats -- no per-dataset
    # generated ZonalStat* models.
    schemas = doc['components']['schemas']
    assert 'CompactStatsResponse' in schemas
    assert not any('ZonalStat' in name for name in schemas)

    # Content negotiation is documented from the format enum: the 200 advertises
    # both the JSON envelope (owned by the response model) and the streamed CSV.
    content = paths['/datasets/{dataset}/stats/{triplet}/date-range']['get'][
        'responses'
    ]['200']['content']
    assert set(content) == {'application/json', 'text/csv'}
    assert content['application/json']['schema']['$ref'].endswith(
        'CompactStatsResponse',
    )
    assert content['text/csv']['schema'] == {'type': 'string'}
