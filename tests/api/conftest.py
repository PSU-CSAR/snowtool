from collections.abc import Iterator

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from snowtool.api.app import get_app


@pytest.fixture
def test_app(test_settings) -> FastAPI:
    # The app builds a SnowDb at startup; a missing on-disk layout is tolerated
    # (it just serves no data and logs a warning), so nothing needs to exist
    # under the configured root here.
    return get_app(settings=test_settings)


@pytest.fixture
def test_client(test_app) -> Iterator[TestClient]:
    with TestClient(test_app) as client:
        yield client
