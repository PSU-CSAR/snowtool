from collections.abc import Iterator

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from snowtool.api.app import get_app


@pytest.fixture
def test_app(test_settings) -> FastAPI:
    # The app builds a SnowDb at startup, which requires a data/ directory under
    # the configured root (empty here -> zero datasets discovered).
    (test_settings.snowdb_path / 'data').mkdir(exist_ok=True)
    return get_app(settings=test_settings)


@pytest.fixture
def test_client(test_app) -> Iterator[TestClient]:
    with TestClient(test_app) as client:
        yield client
