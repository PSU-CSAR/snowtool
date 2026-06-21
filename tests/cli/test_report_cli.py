"""The `report` group and `snowdb validate`, against the synthetic snowdb."""

import json

import numpy

from snowtool.cli import cli
from snowtool.snowdb.cog import write_cog

from ..conftest import SIZE, SWE_VALUE, TILE, snodas_swe_name


def _create(runner, cli_obj, source_dem):
    return runner.invoke(
        cli,
        ['dataset', 'create', 'test', '--dem', str(source_dem)],
        obj=cli_obj,
    )


def _write_swe(root, grid, date_str='20180427'):
    cogs = root / 'data' / 'test' / 'cogs' / date_str
    cogs.mkdir(parents=True, exist_ok=True)
    write_cog(
        cogs / f'{snodas_swe_name(date_str)}.tif',
        numpy.full((SIZE, SIZE), SWE_VALUE, dtype=numpy.int16),
        transform=grid.base_grid.transform,
        tile_size=TILE,
        predictor=2,
    )


# --- report ------------------------------------------------------------------


def test_coverage_reports_gap(runner, cli_obj, initialized_root):
    cogs = initialized_root / 'data' / 'test' / 'cogs'
    for name in ('20180101', '20180103'):
        (cogs / name).mkdir(parents=True)

    result = runner.invoke(cli, ['report', 'coverage', '--format', 'json'], obj=cli_obj)

    row = json.loads(result.output)[0]
    assert row['dates'] == 2
    assert row['gaps'] == 1
    assert row['gap_ranges'] == '2018-01-02..2018-01-02'


def test_missing_files_flags_uncreated_dataset(runner, cli_obj):
    result = runner.invoke(
        cli,
        ['report', 'missing-files', '--format', 'json'],
        obj=cli_obj,
    )

    rows = json.loads(result.output)
    assert rows[0]['dataset'] == 'test'
    assert 'terrain' in rows[0]['missing']


def test_grid_report(runner, cli_obj):
    result = runner.invoke(
        cli,
        ['report', 'grid', 'test', '--format', 'json'],
        obj=cli_obj,
    )

    info = json.loads(result.output)
    assert info['rows'] == 512
    assert info['n_tiles'] == 4
    assert info['is_geographic'] is True


def test_value_ranges_without_dates_errors(runner, cli_obj):
    result = runner.invoke(cli, ['report', 'value-ranges', 'test'], obj=cli_obj)

    assert result.exit_code != 0
    assert 'no ingested dates' in result.output


def test_value_ranges_with_data(runner, cli_obj, source_dem, initialized_root, grid):
    _create(runner, cli_obj, source_dem)
    _write_swe(initialized_root, grid)

    result = runner.invoke(
        cli,
        ['report', 'value-ranges', 'test', '--format', 'json'],
        obj=cli_obj,
    )

    rows = json.loads(result.output)
    swe = next(r for r in rows if r['variable'] == 'swe')
    assert swe['min'] == swe['max'] == 50
    assert swe['nodata_pct'] == 0.0


def test_completeness_reports_missing_variables(runner, cli_obj, initialized_root):
    # A bare date directory with no variable files at all.
    (initialized_root / 'data' / 'test' / 'cogs' / '20180101').mkdir(parents=True)

    result = runner.invoke(
        cli,
        ['report', 'completeness', '--format', 'json'],
        obj=cli_obj,
    )

    rows = json.loads(result.output)
    assert rows[0]['date'] == '2018-01-01'
    assert 'swe' in rows[0]['missing']


def test_aoi_coverage_cli_flags_unrasterized(
    runner,
    cli_obj,
    source_dem,
    initialized_root,
    aoi_geojson,
):
    import shutil

    _create(runner, cli_obj, source_dem)
    shutil.copy(aoi_geojson, initialized_root / 'aois' / 'records' / 'pp.geojson')

    result = runner.invoke(
        cli,
        ['report', 'aoi-coverage', '--format', 'json'],
        obj=cli_obj,
    )

    rows = json.loads(result.output)
    assert rows[0]['triplet'] == '12345:MT:USGS'
    assert rows[0]['issue'] == 'no raster'


def test_aoi_health_cli_clean_when_rasterized(runner, cli_obj, source_dem, aoi_geojson):
    from snowtool.snowdb.aoi import AOI

    _create(runner, cli_obj, source_dem)
    cli_obj.snowdb['test'].rasterize_aoi(AOI.from_geojson(aoi_geojson))

    result = runner.invoke(
        cli,
        ['report', 'aoi-health', '--format', 'json'],
        obj=cli_obj,
    )

    # A healthy raster produces no findings.
    assert json.loads(result.output) == []


# --- snowdb validate ---------------------------------------------------------


def test_validate_ok_on_clean_created_dataset(runner, cli_obj, source_dem):
    _create(runner, cli_obj, source_dem)

    result = runner.invoke(cli, ['snowdb', 'validate'], obj=cli_obj)

    assert result.exit_code == 0
    assert 'ok' in result.output


def test_validate_exits_nonzero_on_missing_files(runner, cli_obj):
    # Dataset not created -> missing terrain/area/cogs/aoi-rasters.
    result = runner.invoke(cli, ['snowdb', 'validate'], obj=cli_obj)

    assert result.exit_code == 1
    assert 'missing-files: test' in result.output
    assert 'problem(s) found' in result.output


def test_validate_rolls_up_completeness_and_aoi_coverage(
    runner,
    cli_obj,
    source_dem,
    initialized_root,
    aoi_geojson,
):
    import shutil

    _create(runner, cli_obj, source_dem)
    # A date dir with no variables (completeness finding).
    (initialized_root / 'data' / 'test' / 'cogs' / '20180101').mkdir(parents=True)
    # A global AOI with no raster (aoi-coverage finding).
    shutil.copy(aoi_geojson, initialized_root / 'aois' / 'records' / 'pp.geojson')

    result = runner.invoke(cli, ['snowdb', 'validate'], obj=cli_obj)

    assert result.exit_code == 1
    assert 'incomplete: test 2018-01-01' in result.output
    assert 'aoi-no-raster: test 12345:MT:USGS' in result.output
