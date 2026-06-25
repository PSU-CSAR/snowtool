"""`snowdb init`: it creates an empty, registered-dataset-free root."""

from snowtool.cli import cli
from snowtool.cli._context import CliContext
from snowtool.snowdb.config import CONFIG_FILENAME, RootConfig
from snowtool.snowdb.db import SnowDb


def _init(runner, root):
    return runner.invoke(
        cli,
        ['snowdb', 'init', str(root)],
        obj=CliContext(config=root),
    )


def test_init_creates_an_empty_layout(runner, tmp_path):
    root = tmp_path / 'db'

    result = _init(runner, root)

    assert result.exit_code == 0, result.output
    assert (root / 'aois' / 'records').is_dir()
    assert (root / 'data').is_dir()
    # The root config exists and registers no datasets -- they are added later.
    config = RootConfig.load(root / CONFIG_FILENAME)
    assert config.datasets == {}
    assert list(SnowDb.open(root)) == []


def test_init_is_idempotent(runner, tmp_path):
    root = tmp_path / 'db'
    first = _init(runner, root)
    created_at = RootConfig.load(root / CONFIG_FILENAME).created_at
    second = _init(runner, root)

    assert first.exit_code == second.exit_code == 0
    # The second run leaves the existing config (and its stamp) untouched.
    assert RootConfig.load(root / CONFIG_FILENAME).created_at == created_at
