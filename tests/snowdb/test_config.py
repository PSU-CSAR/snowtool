"""The resource-typed root config: defaults, round-trip, and validation."""

from pathlib import Path

import pytest

from pydantic import ValidationError

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

    with pytest.raises(ValidationError):
        RootConfig.load(path)
