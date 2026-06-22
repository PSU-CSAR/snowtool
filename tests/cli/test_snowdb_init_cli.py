"""`snowdb init`: layout creation plus area/zone-layer generation and its options.

These reuse the CLI suite's autouse fast-terrain stub (see conftest), so they
exercise the init wiring -- which datasets get generated, --quick, and the
--dataset-source override -- without running the real streaming engine.
"""

from snowtool.cli import cli
from snowtool.cli._context import CliContext
from snowtool.snowdb.dem_source import LocalFile
from snowtool.snowdb.landcover_source import LocalFile as LocalNLCD


def _ctx(root, spec, source_dem, source_nlcd):
    return CliContext(
        root=root,
        specs=(spec,),
        zone_layer_sources={
            'terrain': LocalFile(source_dem),
            'landcover': LocalNLCD(source_nlcd),
        },
    )


def test_init_creates_layout_and_terrain(
    runner,
    tmp_path,
    spec,
    source_dem,
    source_nlcd,
):
    root = tmp_path / 'db'
    result = runner.invoke(
        cli,
        ['snowdb', 'init', str(root)],
        obj=_ctx(root, spec, source_dem, source_nlcd),
    )

    assert result.exit_code == 0
    assert (root / 'aois' / 'records').is_dir()
    assert (root / 'data' / 'test').is_dir()
    # area raster (geographic) + terrain + land-cover sets from the default sources.
    assert (root / 'data' / 'test' / 'areas.tif').is_file()
    assert (root / 'data' / 'test' / 'terrain' / 'elevation.tif').is_file()
    assert (root / 'data' / 'test' / 'landcover' / 'forest_cover_pct.tif').is_file()
    assert 'generating terrain (default source)' in result.output
    assert 'generating landcover (default source)' in result.output


def test_init_quick_skips_generation(runner, tmp_path, spec, source_dem, source_nlcd):
    root = tmp_path / 'db'
    result = runner.invoke(
        cli,
        ['snowdb', 'init', str(root), '--quick'],
        obj=_ctx(root, spec, source_dem, source_nlcd),
    )

    assert result.exit_code == 0
    assert (root / 'data' / 'test').is_dir()
    assert not (root / 'data' / 'test' / 'areas.tif').exists()
    assert not (root / 'data' / 'test' / 'terrain').exists()
    assert not (root / 'data' / 'test' / 'landcover').exists()


def test_init_dataset_terrain_override(runner, tmp_path, spec, source_dem, source_nlcd):
    root = tmp_path / 'db'
    result = runner.invoke(
        cli,
        ['snowdb', 'init', str(root),
         '--dataset-source', 'terrain', 'test', str(source_dem)],
        obj=_ctx(root, spec, source_dem, source_nlcd),
    )

    assert result.exit_code == 0
    assert f'generating terrain for test from {source_dem}' in result.output
    # The override dataset is not in the default-source *terrain* pass, but land
    # cover still comes from the default source.
    assert 'generating terrain (default source)' not in result.output
    assert (root / 'data' / 'test' / 'terrain' / 'elevation.tif').is_file()


def test_init_dataset_landcover_override(
    runner, tmp_path, spec, source_dem, source_nlcd,
):
    root = tmp_path / 'db'
    result = runner.invoke(
        cli,
        ['snowdb', 'init', str(root),
         '--dataset-source', 'landcover', 'test', str(source_nlcd)],
        obj=_ctx(root, spec, source_dem, source_nlcd),
    )

    assert result.exit_code == 0
    assert f'generating landcover for test from {source_nlcd}' in result.output
    assert 'generating landcover (default source)' not in result.output
    assert (root / 'data' / 'test' / 'landcover' / 'forest_cover_pct.tif').is_file()


def test_init_is_idempotent(runner, tmp_path, spec, source_dem, source_nlcd):
    root = tmp_path / 'db'
    ctx = lambda: _ctx(root, spec, source_dem, source_nlcd)  # noqa: E731 - terse reuse
    first = runner.invoke(cli, ['snowdb', 'init', str(root)], obj=ctx())
    second = runner.invoke(cli, ['snowdb', 'init', str(root)], obj=ctx())

    assert first.exit_code == second.exit_code == 0
    # Second run finds terrain + land cover already present, so it generates nothing.
    assert 'generating terrain' not in second.output
    assert 'generating landcover' not in second.output
