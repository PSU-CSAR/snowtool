"""CLI test helpers: a runner plus an initialized synthetic snowdb context.

Commands run against the small synthetic `spec` (top-level conftest), injected
into the CLI via a pre-seeded CliContext on ctx.obj -- the root `cli` group
honors an existing context, so `runner.invoke(cli, args, obj=cli_obj)` drives the
real commands against the tiny grid instead of the full snodas spec.

The real terrain/land-cover engines reproject the source onto a 10 m grid, which
is intractable on the degree-scale synthetic grid. So `cli_obj` injects providers
carrying fast stand-in engines (uniform writes) through the same
``zone_layer_providers`` seam the CLI already uses -- no monkeypatching. The
engines themselves are exercised in test_terrain_generate / test_landcover_generate.
"""

from pathlib import Path

import pytest
import rasterio

from click.testing import CliRunner

from snowtool.cli import _console
from snowtool.cli._context import CliContext
from snowtool.snowdb.config import CONFIG_FILENAME, DATASET_CONFIG_FILENAME, RootConfig
from snowtool.snowdb.datasets import config_from_spec
from snowtool.snowdb.manager import SnowDbManager
from snowtool.snowdb.zones.landcover import LandCoverProvider
from snowtool.snowdb.zones.terrain import TerrainProvider

from ..conftest import (
    register_dataset_config,
    write_uniform_landcover,
    write_uniform_terrain,
)


@pytest.fixture(autouse=True)
def _restore_console():
    """Root-option invocations (e.g. --quiet) mutate the module-global console pair."""
    yield
    _console.configure()


@pytest.fixture(autouse=True)
def _no_ambient_snowdb_config(monkeypatch):
    """Strip a maintainer's exported ``SNOWTOOL_SNOWDB_CONFIG`` for every CLI test.

    ``_apply_config`` (``cli/_context.py``) only ever *sets* ``CliContext.config``
    from the env var -- it no longer special-cases "injected context, ambient env
    var" -- so an env var exported in the shell running the suite would otherwise
    leak into every real (non-injected) ``CliRunner().invoke(cli, ...)`` call here
    (``test_context.py`` covers the injected-context case explicitly, with its own
    ``monkeypatch.setenv``).
    """
    monkeypatch.delenv('SNOWTOOL_SNOWDB_CONFIG', raising=False)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def initialized_root(tmp_path, spec):
    """An initialized snowdb root with the synthetic 'test' dataset registered.

    The dataset is staged + registered (a path link to its ``dataset.json``), so
    opening the root serves 'test' -- the config-path equivalent of the old spec
    injection.
    """
    manager = SnowDbManager.initialize(tmp_path)
    register_dataset_config(manager, 'test', config_from_spec(spec))
    return tmp_path


@pytest.fixture
def stage_test_dataset():
    """Stage the already-registered 'test' dataset (skeleton + AOI rasters).

    ``dataset create`` now only stamps a *new* dataset from ``--template``, so
    staging the synthetic 'test' dataset (registered directly by
    ``initialized_root``, not via a template) goes straight through the same
    manager method the CLI command calls.
    """

    def _stage(cli_obj, initialized_root):
        config_path = initialized_root / 'data' / 'test' / DATASET_CONFIG_FILENAME
        return cli_obj.manager.stage_dataset('test', config_path)

    return _stage


def _target_crs(target):
    """The target grid's CRS as a rasterio CRS (griffine CRS -> WKT -> rasterio)."""
    return rasterio.crs.CRS.from_wkt(target.grid.crs.to_wkt())


def _fake_terrain_engine(
    src,
    targets,
    *,
    work_crs=None,
    work_resolution=None,
    workers=None,
    block_size=None,
    force=False,
    progress=None,
):
    """Stand in for generate_terrain: a uniform terrain set per target.

    Matches the real engine's signature so it drops into ``TerrainProvider`` via
    the ``engine=`` seam; only CLI wiring is under test here.
    """
    return {
        target.name: write_uniform_terrain(
            target.directory,
            base_grid=target.grid.base_grid,
            crs=_target_crs(target),
            tile_size=target.tile_size,
        )
        for target in targets
    }


def _fake_landcover_engine(src, targets, *, force=False):
    """Stand in for generate_landcover: a uniform forest layer per target."""
    return {
        target.name: write_uniform_landcover(
            target.directory,
            base_grid=target.grid.base_grid,
            crs=_target_crs(target),
            tile_size=target.tile_size,
        )
        for target in targets
    }


@pytest.fixture
def cli_obj(initialized_root, source_dem, source_nlcd) -> CliContext:
    """A CliContext over the initialized synthetic snowdb (inject as obj=).

    Providers carry fast stand-in engines (no real reprojection); the
    terrain/land-cover generation sources are declared in the root config (local
    files) so the zone-layer commands never reach 3DEP or the MRLC download.
    """
    config_path = initialized_root / CONFIG_FILENAME
    config = RootConfig.load(config_path)
    config.sources = {'terrain': Path(source_dem), 'landcover': Path(source_nlcd)}
    config.save(config_path)
    return CliContext(
        config=initialized_root,
        zone_layer_providers=(
            TerrainProvider(engine=_fake_terrain_engine),
            LandCoverProvider(engine=_fake_landcover_engine),
        ),
    )
