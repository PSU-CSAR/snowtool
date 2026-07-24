"""Settings -> reader wiring: the read-path knobs reach the app-scoped reader.

The API owns the ``Settings``->domain translation (``_provide_reader``). These
pin that each ``SNOWTOOL_``-prefixed env var is read into its :class:`Settings`
field *and* threaded through the production recipe onto the reader -- so a knob
set in the deployment environment actually reaches the long-running server's
reader, not just the settings object.
"""

import pytest

from snowtool.api.app import _provide_reader
from snowtool.api.settings import Settings
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.raster.tiff_cache import DEFAULT_TIFF_CACHE_SIZE
from snowtool.snowdb.zonal_stats import (
    DEFAULT_MAX_CONCURRENT_RASTERS,
    DEFAULT_MAX_ZONE_CELLS,
)

from ..conftest import init_with_builtins


@pytest.fixture
def catalog(tmp_path) -> SnowDb:
    """An opened catalog over a root initialized with the built-in datasets."""
    init_with_builtins(tmp_path)
    return SnowDb.open(tmp_path)


@pytest.mark.parametrize(
    ('field', 'env_var', 'default'),
    [
        ('tiff_cache_size', 'SNOWTOOL_TIFF_CACHE_SIZE', DEFAULT_TIFF_CACHE_SIZE),
        ('max_zone_cells', 'SNOWTOOL_MAX_ZONE_CELLS', DEFAULT_MAX_ZONE_CELLS),
        (
            'max_concurrent_rasters',
            'SNOWTOOL_MAX_CONCURRENT_RASTERS',
            DEFAULT_MAX_CONCURRENT_RASTERS,
        ),
    ],
)
def test_read_path_field_defaults_and_env(
    monkeypatch,
    tmp_path,
    field,
    env_var,
    default,
):
    # Unset -> the library default; the SNOWTOOL_-prefixed env var overrides it.
    monkeypatch.delenv(env_var, raising=False)
    assert getattr(Settings(snowdb_config=tmp_path), field) == default

    monkeypatch.setenv(env_var, '3')
    assert getattr(Settings(snowdb_config=tmp_path), field) == 3


def test_max_concurrent_rasters_env_reaches_the_reader(monkeypatch, tmp_path, catalog):
    # End-to-end: the SNOWTOOL_-prefixed env var is read into Settings and the
    # production recipe threads it onto the app-scoped reader (the knob was
    # previously dropped -- never passed to SnowDbReader).
    monkeypatch.setenv('SNOWTOOL_MAX_CONCURRENT_RASTERS', '5')
    monkeypatch.setenv('SNOWTOOL_MAX_ZONE_CELLS', '7')
    monkeypatch.setenv('SNOWTOOL_TIFF_CACHE_SIZE', '9')

    reader = _provide_reader(catalog, Settings(snowdb_config=tmp_path))

    assert reader.max_concurrent_rasters == 5
    assert reader.max_zone_cells == 7
    assert reader.cache.info().maxsize == 9
