"""The `aoi` command group, against the synthetic snowdb."""

import json

from snowtool.cli import cli

_POINT = {'type': 'Point', 'coordinates': [-119.45, 44.45]}


def _box(x0=-119.9, y0=44.9, x1=-119.0, y1=44.0):
    """A rectangular Polygon geometry inside the synthetic grid's first tile."""
    return {
        'type': 'Polygon',
        'coordinates': [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }


_POLYGON = _box()


def _write_aoi(directory, triplet, *, with_polygon=True):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{triplet.replace(":", "_")}.geojson'
    properties = {'name': triplet, 'source': 'test', 'active': True, 'basinarea': 5.2}
    if with_polygon:
        feature = {
            'type': 'GeometryCollection',
            'id': triplet,
            'geometries': [_POINT, _POLYGON],
            'properties': properties,
        }
    else:
        feature = {
            'type': 'Feature',
            'id': triplet,
            'geometry': _POINT,
            'properties': properties,
        }
    path.write_text(json.dumps(feature))
    return path


def _create_dataset(runner, cli_obj, source_dem):
    return runner.invoke(
        cli,
        ['dataset', 'create', 'test', '--source', 'terrain', str(source_dem)],
        obj=cli_obj,
    )


# --- import / sync -----------------------------------------------------------


def test_import_file(runner, cli_obj, initialized_root, aoi_geojson):
    result = runner.invoke(cli, ['aoi', 'import', str(aoi_geojson)], obj=cli_obj)

    assert result.exit_code == 0
    assert 'imported 1 AOI(s)' in result.output
    assert (initialized_root / 'aois' / 'records' / '12345_MT_USGS.geojson').is_file()
    assert (initialized_root / 'aois' / 'index.geojson').is_file()


def test_import_dir_reports_and_exits_nonzero_on_invalid(
    runner,
    cli_obj,
    initialized_root,
    tmp_path,
):
    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')
    _write_aoi(src, '22222:MT:USGS', with_polygon=False)
    (src / 'bad.geojson').write_text(json.dumps({'type': 'Nonsense'}))

    result = runner.invoke(cli, ['aoi', 'import', str(src)], obj=cli_obj)

    assert result.exit_code != 0
    assert 'imported 1 AOI(s)' in result.output
    assert 'skipped 1 point-only' in result.output
    assert 'invalid source file(s)' in result.output


def test_sync_refuses_without_prune_to(runner, cli_obj, tmp_path, aoi_geojson):
    runner.invoke(cli, ['aoi', 'import', str(aoi_geojson)], obj=cli_obj)
    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')

    result = runner.invoke(cli, ['aoi', 'sync', str(src)], obj=cli_obj)

    assert result.exit_code != 0
    assert 'would be removed' in result.output


def test_sync_prunes_with_archive(runner, cli_obj, tmp_path, aoi_geojson):
    runner.invoke(cli, ['aoi', 'import', str(aoi_geojson)], obj=cli_obj)
    src = tmp_path / 'src'
    _write_aoi(src, '11111:MT:USGS')
    archive = tmp_path / 'archive'

    result = runner.invoke(
        cli,
        ['aoi', 'sync', str(src), '--prune-to', str(archive)],
        obj=cli_obj,
    )

    assert result.exit_code == 0
    assert 'pruned 1 AOI(s)' in result.output
    assert (archive / '12345_MT_USGS.geojson').is_file()


# --- list / show / dump / reindex / remove -----------------------------------


def test_list_json(runner, cli_obj, aoi_geojson):
    runner.invoke(cli, ['aoi', 'import', str(aoi_geojson)], obj=cli_obj)

    result = runner.invoke(cli, ['aoi', 'list', '--format', 'json'], obj=cli_obj)

    rows = json.loads(result.output)
    assert rows[0]['triplet'] == '12345:MT:USGS'
    assert rows[0]['source'] == 'test'


def test_show(runner, cli_obj, aoi_geojson):
    runner.invoke(cli, ['aoi', 'import', str(aoi_geojson)], obj=cli_obj)

    result = runner.invoke(
        cli,
        ['aoi', 'show', '12345:MT:USGS', '--format', 'json'],
        obj=cli_obj,
    )

    record = json.loads(result.output)
    assert record['triplet'] == '12345:MT:USGS'
    assert len(record['geometry_hash']) == 64


def test_show_missing_errors(runner, cli_obj):
    result = runner.invoke(cli, ['aoi', 'show', '99999:MT:USGS'], obj=cli_obj)
    assert result.exit_code != 0


def test_dump(runner, cli_obj, aoi_geojson, tmp_path):
    runner.invoke(cli, ['aoi', 'import', str(aoi_geojson)], obj=cli_obj)
    out = tmp_path / 'out'

    result = runner.invoke(
        cli,
        ['aoi', 'dump', '12345:MT:USGS', '-o', str(out)],
        obj=cli_obj,
    )

    assert result.exit_code == 0
    assert (out / '12345_MT_USGS.geojson').is_file()


def test_remove_dry_run_then_real(runner, cli_obj, aoi_geojson, initialized_root):
    runner.invoke(cli, ['aoi', 'import', str(aoi_geojson)], obj=cli_obj)
    record = initialized_root / 'aois' / 'records' / '12345_MT_USGS.geojson'

    dry = runner.invoke(
        cli,
        ['aoi', 'remove', '12345:MT:USGS', '--dry-run'],
        obj=cli_obj,
    )
    assert 'would remove' in dry.output
    assert record.is_file()

    real = runner.invoke(cli, ['aoi', 'remove', '12345:MT:USGS'], obj=cli_obj)
    assert real.exit_code == 0
    assert not record.exists()


def test_reindex(runner, cli_obj, aoi_geojson, initialized_root):
    runner.invoke(cli, ['aoi', 'import', str(aoi_geojson)], obj=cli_obj)
    (initialized_root / 'aois' / 'index.geojson').unlink()

    result = runner.invoke(cli, ['aoi', 'reindex'], obj=cli_obj)

    assert result.exit_code == 0
    assert 'reindexed 1 AOI(s)' in result.output


# --- rasterize ---------------------------------------------------------------


def test_rasterize_all_then_skip(runner, cli_obj, source_dem, aoi_geojson):
    _create_dataset(runner, cli_obj, source_dem)
    runner.invoke(cli, ['aoi', 'import', str(aoi_geojson)], obj=cli_obj)

    first = runner.invoke(cli, ['aoi', 'rasterize', '--all'], obj=cli_obj)
    assert first.exit_code == 0
    assert 'built 1' in first.output

    second = runner.invoke(cli, ['aoi', 'rasterize', '--all'], obj=cli_obj)
    assert 'built 0, skipped 1' in second.output


def test_rasterize_requires_triplet_or_all(runner, cli_obj, source_dem):
    _create_dataset(runner, cli_obj, source_dem)
    result = runner.invoke(cli, ['aoi', 'rasterize'], obj=cli_obj)
    assert result.exit_code != 0
    assert 'exactly one of' in result.output
