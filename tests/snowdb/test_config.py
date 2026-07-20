"""The resource-typed root config: defaults, round-trip, and validation."""

from pathlib import Path

import pytest

from snowtool.exceptions import SnowDbConfigError
from snowtool.snowdb.config import CONFIG_FILENAME, PathDatasetLink, RootConfig


def test_create_stamps_defaults():
    config = RootConfig.create()

    assert config.resource == 'snowtool.snowdb/v1'
    assert config.datasets == {}
    assert config.pourpoint_index == Path('pourpoints/index.geojson')
    assert config.pourpoint_records == Path('pourpoints/records')
    assert config.created_at.tzinfo is not None  # stamped in UTC


def test_round_trips_through_disk(tmp_path):
    config = RootConfig.create()
    config.datasets = {
        'swann-800m': PathDatasetLink(path='data/swann-800m/dataset.json'),
        'instarr': PathDatasetLink(path='/mnt/big/instarr/dataset.json'),
    }
    path = tmp_path / CONFIG_FILENAME

    config.save(path)
    loaded = RootConfig.load(path)

    assert loaded == config
    # Persisted as indented JSON with a trailing newline (matches the index style).
    text = path.read_text()
    assert text.endswith('\n')
    assert '\n  "resource"' in text


def test_load_rejects_a_foreign_resource(tmp_path):
    path = tmp_path / CONFIG_FILENAME
    path.write_text(
        '{"resource": "snowtool.snowdb/v2", "created_at": "2026-01-01T00:00:00Z"}',
    )

    # The canonical loader wraps pydantic's ValidationError into a clean,
    # path-naming SnowDbConfigError -- never a raw pydantic traceback.
    with pytest.raises(SnowDbConfigError, match=str(path)):
        RootConfig.load(path)


@pytest.mark.parametrize(
    'content',
    [
        b'{ "resource": "snowtool.snowdb/v1", "created_at": "2024-01',
        b'\xff\xfe not even utf-8 text',
        b'{"resource": "snowtool.snowdb/v1"}',  # valid JSON, missing created_at
    ],
    ids=['truncated_json', 'non_json_bytes', 'valid_json_wrong_shape'],
)
def test_load_wraps_a_malformed_config_as_a_config_error(tmp_path, content):
    path = tmp_path / CONFIG_FILENAME
    path.write_bytes(content)

    with pytest.raises(SnowDbConfigError, match=str(path)):
        RootConfig.load(path)
