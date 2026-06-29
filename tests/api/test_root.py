def test_get_landing_page(test_client) -> None:
    response = test_client.get('/')
    assert response.status_code == 200
    rjson = response.json()
    rels = {link['rel'] for link in rjson['links']}
    # self + root, plus a 'data' link to each of /datasets and /pourpoints.
    assert {'self', 'root', 'data'} <= rels
    hrefs = {link['href'] for link in rjson['links'] if link['rel'] == 'data'}
    assert any(href.endswith('/datasets') for href in hrefs)
    assert any(href.endswith('/pourpoints') for href in hrefs)
    # RootRouter additionally advertises conformance + the OpenAPI doc/docs UI.
    assert {'conformance', 'service-desc', 'service-doc'} <= rels
    # Title/description fall back to the app's (set once in get_app).
    assert rjson['title'] == 'PSU CSAR snowtool API'


def test_conformance_declaration(test_client) -> None:
    response = test_client.get('/conformance')
    assert response.status_code == 200
    classes = set(response.json()['conformsTo'])
    # Baseline derived from the running app: core + landing page + json, and oas30
    # because the app exposes an OpenAPI document.
    assert any(uri.endswith('/core') for uri in classes)
    assert any('oas30' in uri for uri in classes)


def test_problems_catalog(test_client) -> None:
    response = test_client.get('/problems')
    assert response.status_code == 200
    catalog = response.json()
    # Every registered problem key resolves to its stable type URI + status.
    assert catalog['dataset-not-found']['type'] == '/problems/dataset-not-found'
    assert catalog['dataset-not-found']['status'] == 404
    assert catalog['pourpoint-not-covered']['status'] == 409


def test_get_problem_type(test_client) -> None:
    response = test_client.get('/problems/invalid-query-parameter')
    assert response.status_code == 200
    assert response.json()['status'] == 422


def test_get_unknown_problem_type_404(test_client) -> None:
    response = test_client.get('/problems/nope')
    assert response.status_code == 404
    assert response.json()['type'] == '/problems/problem-type-not-found'


def test_get_version(test_client) -> None:
    response = test_client.get('/version')
    assert response.status_code == 200
    rjson = response.json()
    assert 'version' in rjson
    assert {link['rel'] for link in rjson['links']} == {'self', 'root'}
