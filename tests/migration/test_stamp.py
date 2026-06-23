"""Stamping a root config onto a legacy snowdb root."""

import pytest

from snowtool.migration.stamp import stamp_root
from snowtool.snowdb.config import CONFIG_FILENAME, RootConfig
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.spec import DatasetSpec, GridParams


def _spec(name: str) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        grid_params=GridParams(
            origin_x=-120.0,
            origin_y=45.0,
            px_size=0.01,
            cols=256,
            rows=256,
            tile_size=256,
        ),
    )


def _legacy_root(tmp_path):
    """A pre-config snowdb root: the base tree, but no snowdb_conf.json."""
    (tmp_path / 'aois' / 'records').mkdir(parents=True)
    (tmp_path / 'data' / 'snodas').mkdir(parents=True)
    return tmp_path


def test_stamp_writes_a_loadable_config(tmp_path):
    root = _legacy_root(tmp_path)

    config_path, written = stamp_root(root)

    assert written is True
    assert config_path == root / CONFIG_FILENAME
    assert RootConfig.load(config_path).datasets == []


def test_open_succeeds_after_stamping(tmp_path):
    root = _legacy_root(tmp_path)

    stamp_root(root)

    # The whole point: a legacy root that open() refused now opens cleanly.
    db = SnowDb.open(root, [_spec('snodas')])
    assert list(db) == ['snodas']


def test_stamp_is_idempotent(tmp_path):
    root = _legacy_root(tmp_path)
    _, first = stamp_root(root)
    created_at = RootConfig.load(root / CONFIG_FILENAME).created_at

    _, second = stamp_root(root)

    assert first is True
    assert second is False  # left untouched
    assert RootConfig.load(root / CONFIG_FILENAME).created_at == created_at


def test_stamp_rejects_a_non_directory(tmp_path):
    target = tmp_path / 'a-file'
    target.write_text('not a dir')

    with pytest.raises(ValueError, match='Not a directory'):
        stamp_root(target)


def test_stamp_rejects_a_non_snowdb_dir(tmp_path):
    (tmp_path / 'aois').mkdir()  # has aois/ but no data/

    with pytest.raises(ValueError, match='missing data'):
        stamp_root(tmp_path)


def test_cli_migration_stamp(tmp_path):
    from click.testing import CliRunner

    from snowtool.cli import cli

    root = _legacy_root(tmp_path)

    result = CliRunner().invoke(cli, ['migration', 'stamp', str(root)])

    assert result.exit_code == 0, result.output
    assert 'stamped' in result.output
    assert (root / CONFIG_FILENAME).is_file()
