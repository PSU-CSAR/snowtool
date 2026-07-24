"""doctor: the rolled-up health checks (exit 1 on findings, [] means clean)."""

import json
import shutil

import numpy
import rasterio

from snowtool.cli import cli
from snowtool.snowdb.constants import DEM_HASH_TAG
from snowtool.snowdb.pourpoint import Pourpoint
from snowtool.snowdb.raster.cog import write_cog
from snowtool.snowdb.zones.terrain_layers import ELEVATION

from ..conftest import TILE, snodas_swe_name


def _doctor(runner, cli_obj, *args):
    return runner.invoke(cli, ['doctor', '--format', 'json', *args], obj=cli_obj)


def _create(runner, cli_obj, source_dem, initialized_root, stage_test_dataset):
    """Stage the synthetic dataset, then generate its zone layers explicitly."""
    stage_test_dataset(cli_obj, initialized_root)
    generated = runner.invoke(
        cli,
        ['dataset', 'generate-zones', 'test', '--source', 'terrain', str(source_dem)],
        obj=cli_obj,
    )
    assert generated.exit_code == 0, generated.output
    return generated


def test_clean_db_exits_zero_with_empty_findings(
    runner,
    cli_obj,
    source_dem,
    initialized_root,
    stage_test_dataset,
):
    _create(runner, cli_obj, source_dem, initialized_root, stage_test_dataset)

    result = _doctor(runner, cli_obj)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []


def test_unknown_check_is_a_clean_usage_error(runner, cli_obj):
    result = _doctor(runner, cli_obj, 'nonsense')

    assert result.exit_code != 0
    assert 'grid, dates, files, pourpoints' in result.output


def test_missing_cogs_yields_files_finding_and_exit_1(runner, cli_obj):
    # Dataset not created -> missing terrain/cogs/aoi-rasters.
    result = _doctor(runner, cli_obj, 'files')

    assert result.exit_code == 1
    findings = json.loads(result.stdout)
    assert findings
    assert all(f['check'] == 'files' for f in findings)
    assert any(f['target'] == 'cogs' for f in findings)


def test_check_selection_limits_the_sweep(runner, cli_obj):
    # With the same broken/uncreated-dataset arrangement, `doctor grid` stays clean.
    result = _doctor(runner, cli_obj, 'grid')

    assert result.exit_code == 0
    assert json.loads(result.stdout) == []


def test_dates_check_reports_missing_variables(runner, cli_obj, initialized_root):
    # A bare date directory with no variable files at all.
    (initialized_root / 'data' / 'test' / 'cogs' / '20180101').mkdir(parents=True)

    result = _doctor(runner, cli_obj, 'dates')

    findings = json.loads(result.stdout)
    assert findings[0]['check'] == 'dates'
    assert findings[0]['dataset'] == 'test'
    assert findings[0]['target'] == '2018-01-01'
    assert 'swe' in findings[0]['issue']


def test_pourpoints_check_flags_unrasterized(
    runner,
    cli_obj,
    source_dem,
    initialized_root,
    pourpoint_geojson,
    stage_test_dataset,
):
    _create(runner, cli_obj, source_dem, initialized_root, stage_test_dataset)
    shutil.copy(
        pourpoint_geojson,
        initialized_root / 'pourpoints' / 'records' / '12345_MT_USGS.geojson',
    )

    result = _doctor(runner, cli_obj, 'pourpoints')

    findings = json.loads(result.stdout)
    assert findings[0]['check'] == 'pourpoints'
    assert findings[0]['target'] == '12345:MT:USGS'
    assert findings[0]['issue'] == 'no raster'


def test_pourpoints_check_clean_when_rasterized(
    runner,
    cli_obj,
    source_dem,
    initialized_root,
    pourpoint_geojson,
    stage_test_dataset,
):
    _create(runner, cli_obj, source_dem, initialized_root, stage_test_dataset)
    # The pourpoint must be registered (not just rasterized), or the raster
    # counts as an orphan against the registry.
    shutil.copy(
        pourpoint_geojson,
        initialized_root / 'pourpoints' / 'records' / '12345_MT_USGS.geojson',
    )
    cli_obj.snowdb['test'].rasterize_aoi(Pourpoint.from_geojson(pourpoint_geojson))

    result = _doctor(runner, cli_obj, 'pourpoints')

    # A healthy, registered, rasterized pourpoint produces no findings.
    assert json.loads(result.stdout) == []


def test_skips_inactive_by_default(runner, cli_obj):
    # 'test' is registered but uncreated (missing files -> findings). Deactivated,
    # it drops out of the default sweep: doctor gates what readers serve, so a
    # half-built inactive dataset must not fail cron/CI. Each invocation gets a
    # fresh CliContext (a real CLI process opens the root config anew).
    def ctx():
        from snowtool.cli._context import CliContext

        return CliContext(
            config=cli_obj.config,
            zone_layer_providers=cli_obj.zone_layer_providers,
        )

    deactivated = runner.invoke(cli, ['dataset', 'deactivate', 'test'], obj=ctx())
    assert deactivated.exit_code == 0, deactivated.output

    result = _doctor(runner, ctx())
    assert result.exit_code == 0
    assert json.loads(result.stdout) == []

    # --include-inactive widens the sweep back to everything registered ...
    widened = _doctor(runner, ctx(), '--include-inactive')
    assert widened.exit_code == 1
    findings = json.loads(widened.stdout)
    assert any(f['check'] == 'files' and f['dataset'] == 'test' for f in findings)

    # ... and an explicit -d NAME always resolves from registered.
    named = _doctor(runner, ctx(), '-d', 'test')
    assert named.exit_code == 1
    findings = json.loads(named.stdout)
    assert any(f['check'] == 'files' and f['dataset'] == 'test' for f in findings)


def test_rolls_up_dates_and_pourpoint_findings(
    runner,
    cli_obj,
    source_dem,
    initialized_root,
    pourpoint_geojson,
    stage_test_dataset,
):
    _create(runner, cli_obj, source_dem, initialized_root, stage_test_dataset)
    # A date dir with no variables (dates finding).
    (initialized_root / 'data' / 'test' / 'cogs' / '20180101').mkdir(parents=True)
    # A pourpoint with no AOI raster (pourpoints finding).
    shutil.copy(
        pourpoint_geojson,
        initialized_root / 'pourpoints' / 'records' / '12345_MT_USGS.geojson',
    )

    result = _doctor(runner, cli_obj)

    assert result.exit_code == 1
    findings = json.loads(result.stdout)
    assert any(
        f['check'] == 'dates' and f['target'] == '2018-01-01' and 'swe' in f['issue']
        for f in findings
    )
    assert any(
        f['check'] == 'pourpoints'
        and f['target'] == '12345:MT:USGS'
        and f['issue'] == 'no raster'
        for f in findings
    )


def test_grid_check_flags_grid_drift(runner, cli_obj, initialized_root, grid):
    # A COG whose shape does not match the declared grid surfaces as a finding.
    cogs = initialized_root / 'data' / 'test' / 'cogs' / '20180101'
    cogs.mkdir(parents=True)
    write_cog(
        cogs / f'{snodas_swe_name("20180101")}.tif',
        numpy.zeros((256, 256), dtype=numpy.int16),
        transform=grid.base_grid.transform,
        tile_size=TILE,
    )

    result = _doctor(runner, cli_obj, 'grid')

    assert result.exit_code == 1
    findings = json.loads(result.stdout)
    assert findings
    assert all(f['check'] == 'grid' and f['dataset'] == 'test' for f in findings)


def test_files_check_flags_stale_zone_layer_format(
    runner,
    cli_obj,
    source_dem,
    initialized_root,
    stage_test_dataset,
):
    # Re-stamp the built terrain set with an old format version (what an artifact
    # from before a format bump would carry) -> doctor flags it for a rebuild.
    _create(runner, cli_obj, source_dem, initialized_root, stage_test_dataset)
    elevation = initialized_root / 'data' / 'test' / 'terrain' / ELEVATION.filename
    with rasterio.open(
        elevation,
        'r+',
        IGNORE_COG_LAYOUT_BREAK='YES',
    ) as ds:
        ds.update_tags(**{DEM_HASH_TAG: 'v999:deadbeef'})

    result = _doctor(runner, cli_obj, 'files')

    assert result.exit_code == 1
    findings = json.loads(result.stdout)
    stale = next(f for f in findings if f['target'] == 'terrain')
    assert stale['check'] == 'files'
    assert stale['dataset'] == 'test'
    assert 'stored 999 != current 3' in stale['issue']
