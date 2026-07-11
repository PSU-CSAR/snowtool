"""The `report` group, against the synthetic snowdb."""

import json

import numpy

from snowtool.cli import cli
from snowtool.snowdb.raster.cog import write_cog

from ..conftest import SIZE, SWE_VALUE, TILE, snodas_swe_name


def _create(runner, cli_obj, source_dem):
    """Stage the synthetic dataset, then generate its zone layers explicitly
    (create is stage-only; zones come only from generate-zones)."""
    result = runner.invoke(cli, ['dataset', 'create', 'test'], obj=cli_obj)
    assert result.exit_code == 0, result.output
    generated = runner.invoke(
        cli,
        ['dataset', 'generate-zones', 'test', '--source', 'terrain', str(source_dem)],
        obj=cli_obj,
    )
    assert generated.exit_code == 0, generated.output
    return generated


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
