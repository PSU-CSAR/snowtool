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
from snowtool.snowdb import datasets as datasets_mod
from snowtool.snowdb.config import CONFIG_FILENAME, DATASET_CONFIG_FILENAME, RootConfig
from snowtool.snowdb.datasets import config_from_spec
from snowtool.snowdb.manager import SnowDbManager
from snowtool.snowdb.zones.landcover import landcover_provider
from snowtool.snowdb.zones.terrain import terrain_provider

from ..conftest import (
    register_dataset_config,
    write_marker_cog,
    write_uniform_landcover,
    write_uniform_terrain,
)


class _RecordingIngester:
    """A fake ``Ingester`` whose per-run ``plan`` is supplied by the test.

    Records each ``plan`` call's ``(source, dataset.spec.name)`` so a test can
    assert the driver invoked it once (and with what source), while delegating the
    actual ``DateIngest`` sequence to the injected ``plan_fn`` -- the only part
    that differs between the ingest CLI tests.
    """

    def __init__(self, plan_fn):
        self._plan_fn = plan_fn
        self.calls = []

    def plan(self, source, dataset):
        self.calls.append((source, dataset.spec.name))
        yield from self._plan_fn(source, dataset)


@pytest.fixture
def register_fake_ingester(monkeypatch, tmp_path, spec):
    """Register a fake 'fake' ingester on the synthetic 'test' dataset.

    Collapses the repeated ingest-CLI setup (register 'fake' in the INGESTERS
    map, initialize the root, stage a spec config naming it) into one seam: call
    it with a ``plan_fn(source, dataset)`` generator and it returns the wired-up
    :class:`_RecordingIngester`. The INGESTERS monkeypatch happens here, once.
    """

    def _register(plan_fn):
        fake = _RecordingIngester(plan_fn)
        monkeypatch.setitem(datasets_mod.INGESTERS, 'fake', fake)
        manager = SnowDbManager.initialize(tmp_path)
        config = config_from_spec(spec)
        config.ingester = 'fake'
        register_dataset_config(manager, 'test', config)
        return fake

    return _register


class FakeRaster:
    """A ``WritableRaster`` that drops a tiny real marker COG into the date dir.

    ``out_name`` is the filename the COG lands under; it carries ``source_hash`` in
    its ``SOURCE_HASH`` tag (what a real ingester stamps, and what the per-date skip
    check reads back), so a fake ingester built on these drives the genuine atomic
    ``Dataset.write_date_cogs`` path end-to-end.
    """

    def __init__(self, out_name: str, source_hash: str) -> None:
        self.out_name = out_name
        self.source_hash = source_hash

    def write_cog(self, output_dir) -> None:
        write_marker_cog(output_dir / self.out_name, self.source_hash)


def _name_for_glob(glob: str) -> str:
    """A concrete filename that matches ``glob`` (``*`` dropped, ``?``/``[..]`` pinned).

    Turns a variable's ``fnmatch`` glob into one deterministic filename: ``*`` -> "",
    ``?`` -> "0", ``[ab..]`` -> its first char. Distinct variable globs (which differ
    by product code) yield distinct names, so a full set resolves one COG per variable.
    """
    out: list[str] = []
    i = 0
    while i < len(glob):
        char = glob[i]
        if char == '*':
            i += 1
        elif char == '?':
            out.append('0')
            i += 1
        elif char == '[':
            close = glob.index(']', i)
            out.append(glob[i + 1])
            i = close + 1
        else:
            out.append(char)
            i += 1
    return ''.join(out)


def full_marker_out_names(dataset) -> frozenset[str]:
    """The COG filenames :func:`full_marker_rasters` will land, one per spec variable.

    A fake ingester's ``DateIngest.out_names``: the write path's per-date skip check
    reads these (with the source hash) *before* ``build_rasters`` runs, so they must
    match what that build produces.
    """
    return frozenset(
        _name_for_glob(variable.glob) for variable in dataset.spec.variables.values()
    )


def full_marker_rasters(dataset, source_hash: str) -> list[FakeRaster]:
    """One :class:`FakeRaster` per spec variable, covering every required variable.

    A fake ingester's ``build_rasters`` returns this so the real ``write_date_cogs``
    completeness check (every variable must resolve to exactly one COG) passes on a
    full SNODAS spec without a real archive.
    """
    return [
        FakeRaster(_name_for_glob(variable.glob), source_hash)
        for variable in dataset.spec.variables.values()
    ]


@pytest.fixture(autouse=True)
def _restore_console():
    """Root-option invocations (e.g. --quiet) mutate the module-global console pair."""
    yield
    _console.configure()


@pytest.fixture(autouse=True)
def _no_ambient_snowdb_config(monkeypatch):
    """Strip a maintainer's exported ``SNOWTOOL_SNOWDB_CONFIG`` for every CLI test.

    ``_apply_config`` (``cli/_context.py``) only ever *sets* ``CliContext.config``
    from the env var, so an env var exported in the shell running the suite
    would otherwise leak into every real (non-injected) ``CliRunner().invoke(cli,
    ...)`` call here (``test_context.py`` covers the injected-context case
    explicitly, with its own ``monkeypatch.setenv``).
    """
    monkeypatch.delenv('SNOWTOOL_SNOWDB_CONFIG', raising=False)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def initialized_root(tmp_path, spec):
    """An initialized snowdb root with the synthetic 'test' dataset registered.

    The dataset is registered via a path link to its ``dataset.json``, so
    opening the root serves 'test'.
    """
    manager = SnowDbManager.initialize(tmp_path)
    register_dataset_config(manager, 'test', config_from_spec(spec))
    return tmp_path


@pytest.fixture
def stage_test_dataset():
    """Stage the already-registered 'test' dataset (skeleton + AOI rasters).

    ``dataset create`` only stamps a *new* dataset from ``--template``, so
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
    source,
    targets,
    bounds,
    *,
    workers=None,
    block_size=None,
    force=False,
    progress=None,
):
    """Stand in for generate_terrain: a uniform terrain set per target.

    Matches the real engine's ``(source, targets, bounds, *, ...)`` signature so it
    drops into the terrain provider via the ``engine=`` seam; it never opens the
    source, so only CLI wiring is under test here.
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


def _fake_landcover_engine(
    source,
    targets,
    bounds,
    *,
    workers=None,
    block_size=None,
    force=False,
    progress=None,
):
    """Stand in for generate_landcover: a uniform forest layer per target.

    Matches the real engine's ``(source, targets, bounds, *, ...)`` signature so it
    drops into the land-cover provider via the ``engine=`` seam; it never opens the
    source, so only CLI wiring is under test here.
    """
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
            terrain_provider(engine=_fake_terrain_engine),
            landcover_provider(engine=_fake_landcover_engine),
        ),
    )
