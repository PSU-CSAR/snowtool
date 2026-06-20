import pytest

from snowtool.settings import Settings


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    return Settings(rasterdb_path=tmp_path)
