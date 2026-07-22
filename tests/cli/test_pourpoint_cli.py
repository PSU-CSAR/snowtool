"""The `pourpoint` command group, against the synthetic snowdb."""

import json

from snowtool.cli import cli

from ..conftest import write_aoi_record as _write_aoi

# --- import / sync -----------------------------------------------------------


def test_import_file(runner, cli_obj, initialized_root, pourpoint_geojson):
    result = runner.invoke(
        cli,
        ['pourpoint', 'import', str(pourpoint_geojson)],
        obj=cli_obj,
    )

    assert result.exit_code == 0
    assert 'imported 1 pourpoint(s)' in result.output
    assert (
        initialized_root / 'pourpoints' / 'records' / '12345_MT_USGS.geojson'
    ).is_file()
    assert (initialized_root / 'pourpoints' / 'index.geojson').is_file()


def test_import_invalid_file_exits_nonzero(runner, cli_obj, tmp_path):
    bad = tmp_path / 'bad.geojson'
    bad.write_text(json.dumps({'type': 'Nonsense'}))

    result = runner.invoke(cli, ['pourpoint', 'import', str(bad)], obj=cli_obj)

    assert result.exit_code != 0
    assert 'imported 0 pourpoint(s)' in result.output
    assert 'invalid source file(s)' in result.output


def test_import_rejects_directory(runner, cli_obj, tmp_path):
    # import is single-record; a directory is a `sync` job.
    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')

    result = runner.invoke(cli, ['pourpoint', 'import', str(src)], obj=cli_obj)

    assert result.exit_code != 0
    assert 'single file' in result.output
    assert 'sync' in result.output


def test_sync_refuses_without_prune_to(runner, cli_obj, tmp_path, pourpoint_geojson):
    runner.invoke(cli, ['pourpoint', 'import', str(pourpoint_geojson)], obj=cli_obj)
    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')

    result = runner.invoke(cli, ['pourpoint', 'sync', str(src)], obj=cli_obj)

    assert result.exit_code != 0
    assert 'would be removed' in result.output


def test_sync_prunes_with_archive(runner, cli_obj, tmp_path, pourpoint_geojson):
    runner.invoke(cli, ['pourpoint', 'import', str(pourpoint_geojson)], obj=cli_obj)
    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')
    archive = tmp_path / 'archive'

    result = runner.invoke(
        cli,
        ['pourpoint', 'sync', str(src), '--prune-to', str(archive)],
        obj=cli_obj,
    )

    assert result.exit_code == 0
    assert 'pruned 1 pourpoint(s)' in result.output
    assert (archive / '12345_MT_USGS.geojson').is_file()


# --- list / show / dump / reindex / remove -----------------------------------


def test_list_json(runner, cli_obj, pourpoint_geojson):
    runner.invoke(cli, ['pourpoint', 'import', str(pourpoint_geojson)], obj=cli_obj)

    result = runner.invoke(cli, ['pourpoint', 'list', '--format', 'json'], obj=cli_obj)

    rows = json.loads(result.output)
    assert rows[0]['triplet'] == '12345:MT:USGS'
    assert rows[0]['area_meters'] > 0


def test_list_table_flattens_coverage(runner, cli_obj, pourpoint_geojson):
    # The coverage cell is a {dataset: Coverage} dict; table/csv must flatten it
    # (key=value) rather than print the raw Python repr.
    runner.invoke(cli, ['pourpoint', 'import', str(pourpoint_geojson)], obj=cli_obj)

    result = runner.invoke(cli, ['pourpoint', 'list', '--format', 'table'], obj=cli_obj)

    assert result.exit_code == 0
    assert 'test=full' in result.output
    assert "{'test'" not in result.output


def test_list_csv_flattens_coverage(runner, cli_obj, pourpoint_geojson):
    runner.invoke(cli, ['pourpoint', 'import', str(pourpoint_geojson)], obj=cli_obj)

    result = runner.invoke(cli, ['pourpoint', 'list', '--format', 'csv'], obj=cli_obj)

    assert result.exit_code == 0
    assert 'test=full' in result.output
    assert "{'test'" not in result.output


def test_show(runner, cli_obj, pourpoint_geojson):
    runner.invoke(cli, ['pourpoint', 'import', str(pourpoint_geojson)], obj=cli_obj)

    result = runner.invoke(
        cli,
        ['pourpoint', 'show', '12345:MT:USGS', '--format', 'json'],
        obj=cli_obj,
    )

    record = json.loads(result.output)
    assert record['triplet'] == '12345:MT:USGS'
    assert len(record['geometry_hash']) == 64


def test_show_missing_errors(runner, cli_obj):
    result = runner.invoke(cli, ['pourpoint', 'show', '99999:MT:USGS'], obj=cli_obj)
    assert result.exit_code != 0


def test_dump(runner, cli_obj, pourpoint_geojson, tmp_path):
    runner.invoke(cli, ['pourpoint', 'import', str(pourpoint_geojson)], obj=cli_obj)
    out = tmp_path / 'out'

    result = runner.invoke(
        cli,
        ['pourpoint', 'dump', '12345:MT:USGS', '-o', str(out)],
        obj=cli_obj,
    )

    assert result.exit_code == 0
    assert (out / '12345_MT_USGS.geojson').is_file()


def test_remove_dry_run_then_real(runner, cli_obj, pourpoint_geojson, initialized_root):
    runner.invoke(cli, ['pourpoint', 'import', str(pourpoint_geojson)], obj=cli_obj)
    record = initialized_root / 'pourpoints' / 'records' / '12345_MT_USGS.geojson'

    dry = runner.invoke(
        cli,
        ['pourpoint', 'remove', '12345:MT:USGS', '--dry-run'],
        obj=cli_obj,
    )
    assert 'would remove' in dry.output
    assert record.is_file()

    real = runner.invoke(
        cli,
        ['pourpoint', 'remove', '12345:MT:USGS', '--yes'],
        obj=cli_obj,
    )
    assert real.exit_code == 0
    assert not record.exists()


def test_remove_without_yes_refuses(
    runner,
    cli_obj,
    pourpoint_geojson,
    initialized_root,
):
    # CliRunner's stdin is not a TTY, so this must fail loudly rather than hang
    # or silently proceed -- the same non-interactive case as CI.
    runner.invoke(cli, ['pourpoint', 'import', str(pourpoint_geojson)], obj=cli_obj)
    record = initialized_root / 'pourpoints' / 'records' / '12345_MT_USGS.geojson'

    result = runner.invoke(cli, ['pourpoint', 'remove', '12345:MT:USGS'], obj=cli_obj)

    assert result.exit_code != 0
    assert '--yes' in result.output
    assert record.is_file()


def test_reindex(runner, cli_obj, pourpoint_geojson, initialized_root):
    runner.invoke(cli, ['pourpoint', 'import', str(pourpoint_geojson)], obj=cli_obj)
    (initialized_root / 'pourpoints' / 'index.geojson').unlink()

    result = runner.invoke(cli, ['pourpoint', 'reindex'], obj=cli_obj)

    assert result.exit_code == 0
    assert 'reindexed 1 pourpoint(s)' in result.output


# --- rasterize ---------------------------------------------------------------


def test_rasterize_all_then_skip(runner, cli_obj, pourpoint_geojson):
    runner.invoke(cli, ['pourpoint', 'import', str(pourpoint_geojson)], obj=cli_obj)

    first = runner.invoke(
        cli,
        ['pourpoint', 'rasterize', '--all', '--format', 'json'],
        obj=cli_obj,
    )
    assert first.exit_code == 0
    rows = json.loads(first.stdout)
    assert rows == [{'triplet': '12345:MT:USGS', 'dataset': 'test', 'action': 'built'}]
    assert 'built 1, skipped 0' in first.stderr

    second = runner.invoke(
        cli,
        ['pourpoint', 'rasterize', '--all', '--format', 'json'],
        obj=cli_obj,
    )
    assert second.exit_code == 0
    rows = json.loads(second.stdout)
    assert rows == [
        {'triplet': '12345:MT:USGS', 'dataset': 'test', 'action': 'skipped'},
    ]
    assert 'built 0, skipped 1' in second.stderr


def test_rasterize_requires_triplet_or_all(runner, cli_obj):
    result = runner.invoke(cli, ['pourpoint', 'rasterize'], obj=cli_obj)
    assert result.exit_code != 0
    assert 'exactly one of' in result.output
