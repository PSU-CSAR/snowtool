"""Root-level snow-database commands: `status` and `init`."""

import json

from snowtool.cli import cli
from snowtool.cli._context import CliContext
from snowtool.snowdb.config import CONFIG_FILENAME, DATASET_CONFIG_FILENAME, RootConfig
from snowtool.snowdb.db import SnowDb


def test_status_json_for_uncreated_dataset(runner, cli_obj):
    result = runner.invoke(cli, ['status', '--format', 'json'], obj=cli_obj)

    assert result.exit_code == 0
    rows = json.loads(result.output)
    assert len(rows) == 1
    row = rows[0]
    assert row['dataset'] == 'test'
    assert row['terrain'] is False
    assert row['dates'] == 0


def test_status_reflects_created_dataset(runner, cli_obj, source_dem, initialized_root):
    # 'test' is registered directly (not via a template), so staging goes
    # straight through the manager method 'dataset create' calls -- that
    # command itself now only stamps a brand-new dataset from --template.
    # Zone layers still come only from the explicit generate-zones pass.
    config_path = initialized_root / 'data' / 'test' / DATASET_CONFIG_FILENAME
    cli_obj.manager.stage_dataset('test', config_path)
    runner.invoke(
        cli,
        ['dataset', 'generate-zones', 'test', '--source', 'terrain', str(source_dem)],
        obj=cli_obj,
    )

    result = runner.invoke(cli, ['status', '--format', 'json'], obj=cli_obj)

    row = json.loads(result.output)[0]
    assert row['terrain'] is True
    assert row['cogs'] is True
    assert 'area' not in row  # no area raster is tracked; the AOI raster carries it


def test_status_table_smoke(runner, cli_obj):
    result = runner.invoke(cli, ['status'], obj=cli_obj)

    assert result.exit_code == 0
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    header, *rows = lines
    assert 'dataset' in header
    assert 'active' in header
    assert any('test' in ln for ln in rows)


def _init(runner, root):
    return runner.invoke(
        cli,
        ['init', str(root)],
        obj=CliContext(config=root),
    )


def test_init_creates_an_empty_layout(runner, tmp_path):
    root = tmp_path / 'db'

    result = _init(runner, root)

    assert result.exit_code == 0, result.output
    assert (root / 'pourpoints' / 'records').is_dir()
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
