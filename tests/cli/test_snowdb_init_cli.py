"""`snowdb init`: layout creation plus area/terrain generation and its options.

These reuse the CLI suite's autouse fast-terrain stub (see conftest), so they
exercise the init wiring -- which datasets get generated, --quick, and the
--dataset-dem override -- without running the real streaming engine.
"""

from snowtool.cli import cli
from snowtool.cli._context import CliContext
from snowtool.snowdb.dem_source import LocalFile


def _ctx(root, spec, source_dem):
    return CliContext(root=root, specs=(spec,), dem_source=LocalFile(source_dem))


def test_init_creates_layout_and_terrain(runner, tmp_path, spec, source_dem):
    root = tmp_path / 'db'
    result = runner.invoke(
        cli,
        ['snowdb', 'init', str(root)],
        obj=_ctx(root, spec, source_dem),
    )

    assert result.exit_code == 0
    assert (root / 'aois' / 'records').is_dir()
    assert (root / 'data' / 'test').is_dir()
    # area raster (geographic) + terrain set generated from the default source.
    assert (root / 'data' / 'test' / 'areas.tif').is_file()
    assert (root / 'data' / 'test' / 'terrain' / 'elevation.tif').is_file()
    assert 'generating terrain (default source)' in result.output


def test_init_quick_skips_area_and_terrain(runner, tmp_path, spec, source_dem):
    root = tmp_path / 'db'
    result = runner.invoke(
        cli,
        ['snowdb', 'init', str(root), '--quick'],
        obj=_ctx(root, spec, source_dem),
    )

    assert result.exit_code == 0
    assert (root / 'data' / 'test').is_dir()
    assert not (root / 'data' / 'test' / 'areas.tif').exists()
    assert not (root / 'data' / 'test' / 'terrain').exists()


def test_init_dataset_dem_override(runner, tmp_path, spec, source_dem):
    root = tmp_path / 'db'
    result = runner.invoke(
        cli,
        ['snowdb', 'init', str(root), '--dataset-dem', 'test', str(source_dem)],
        obj=_ctx(root, spec, source_dem),
    )

    assert result.exit_code == 0
    assert f'generating terrain for test from {source_dem}' in result.output
    # The override dataset is not in the default-source pass.
    assert 'default source' not in result.output
    assert (root / 'data' / 'test' / 'terrain' / 'elevation.tif').is_file()


def test_init_is_idempotent(runner, tmp_path, spec, source_dem):
    root = tmp_path / 'db'
    ctx = lambda: _ctx(root, spec, source_dem)  # noqa: E731 - terse fixture reuse
    first = runner.invoke(cli, ['snowdb', 'init', str(root)], obj=ctx())
    second = runner.invoke(cli, ['snowdb', 'init', str(root)], obj=ctx())

    assert first.exit_code == second.exit_code == 0
    # Second run finds terrain already present, so it generates nothing.
    assert 'generating terrain' not in second.output
