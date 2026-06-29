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


def test_get_version(test_client) -> None:
    response = test_client.get('/version')
    assert response.status_code == 200
    rjson = response.json()
    assert 'version' in rjson
    assert {link['rel'] for link in rjson['links']} == {'self', 'root'}
