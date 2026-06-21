"""The `snowdb status` overview command."""

import json

from snowtool.cli import cli


def test_status_json_for_uncreated_dataset(runner, cli_obj):
    result = runner.invoke(cli, ['snowdb', 'status', '--format', 'json'], obj=cli_obj)

    assert result.exit_code == 0
    rows = json.loads(result.output)
    assert len(rows) == 1
    row = rows[0]
    assert row['dataset'] == 'test'
    assert row['dem'] is False
    assert row['dates'] == 0


def test_status_reflects_created_dataset(runner, cli_obj, source_dem):
    runner.invoke(
        cli,
        ['dataset', 'create', 'test', '--dem', str(source_dem)],
        obj=cli_obj,
    )

    result = runner.invoke(cli, ['snowdb', 'status', '--format', 'json'], obj=cli_obj)

    row = json.loads(result.output)[0]
    assert row['dem'] is True
    assert row['area'] is True  # geographic grid has an area raster
    assert row['cogs'] is True


def test_status_table_smoke(runner, cli_obj):
    result = runner.invoke(cli, ['snowdb', 'status'], obj=cli_obj)

    assert result.exit_code == 0
    assert 'dataset' in result.output
    assert 'test' in result.output
