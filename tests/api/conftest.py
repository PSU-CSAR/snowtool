from collections.abc import Iterator

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from snowtool.api.app import get_app

from ..conftest import init_with_builtins, populate_synthetic_root


@pytest.fixture
def test_app(test_settings) -> FastAPI:
    # The app opens a SnowDb from the root config at startup, so the root must be
    # initialized with the built-in datasets registered for it to serve them.

    init_with_builtins(test_settings.snowdb_config)
    return get_app(settings=test_settings)


@pytest.fixture
def test_client(test_app) -> Iterator[TestClient]:
    with TestClient(test_app) as client:
        yield client


@pytest.fixture
def synthetic_settings(test_settings, spec, aoi_geojson):
    """Settings over a synthetic root populated end-to-end for the 'test' dataset.

    The Settings ``snowdb_config`` seam *is* the injection point: ``get_app`` opens
    its catalog from there and builds the (fresh-per-app) ``SnowDbReader`` over it,
    so a per-test app reads exactly this synthetic snowdb with no monkeypatching.
    """
    populate_synthetic_root(test_settings.snowdb_config, spec, aoi_geojson)
    return test_settings


@pytest.fixture
def synthetic_client(synthetic_settings) -> Iterator[TestClient]:
    with TestClient(get_app(settings=synthetic_settings)) as client:
        yield client
