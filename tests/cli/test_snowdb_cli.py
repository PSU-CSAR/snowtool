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
    assert row['terrain'] is False
    assert row['dates'] == 0


def test_status_reflects_created_dataset(runner, cli_obj, source_dem):
    # Create is stage-only; the zone layers status reports come from the
    # explicit generate-zones pass.
    runner.invoke(cli, ['dataset', 'create', 'test'], obj=cli_obj)
    runner.invoke(
        cli,
        ['dataset', 'generate-zones', 'test', '--source', 'terrain', str(source_dem)],
        obj=cli_obj,
    )

    result = runner.invoke(cli, ['snowdb', 'status', '--format', 'json'], obj=cli_obj)

    row = json.loads(result.output)[0]
    assert row['terrain'] is True
    assert row['cogs'] is True
    assert 'area' not in row  # no area raster is tracked; the AOI raster carries it


def test_status_table_smoke(runner, cli_obj):
    result = runner.invoke(cli, ['snowdb', 'status'], obj=cli_obj)

    assert result.exit_code == 0
    assert 'dataset' in result.output
    assert 'test' in result.output
