from collections.abc import Iterator

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from snowtool.api.app import get_app


@pytest.fixture
def test_app(test_settings) -> FastAPI:
    # The app opens a SnowDb from the root config at startup, so the root must be
    # initialized with the built-in datasets registered for it to serve them.
    from ..conftest import init_with_builtins

    init_with_builtins(test_settings.snowdb_config)
    return get_app(settings=test_settings)


@pytest.fixture
def test_client(test_app) -> Iterator[TestClient]:
    with TestClient(test_app) as client:
        yield client
