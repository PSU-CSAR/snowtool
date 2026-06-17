def test_get_landing_page(test_client) -> None:
    response = test_client.get('/')
    assert response.status_code == 200
    rjson = response.json()
    assert 'links' in rjson
    assert rjson['links'][0]['rel'] == 'root'
    assert rjson['links'][1]['rel'] == 'self'


def test_get_version(test_client) -> None:
    response = test_client.get('/version')
    assert response.status_code == 200
    rjson = response.json()
    assert 'version' in rjson
