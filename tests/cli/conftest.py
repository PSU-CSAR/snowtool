"""CLI test helpers: a runner plus an initialized synthetic snowdb context.

Commands run against the small synthetic `spec` (top-level conftest), injected
into the CLI via a pre-seeded CliContext on ctx.obj -- the root `cli` group
honors an existing context, so `runner.invoke(cli, args, obj=cli_obj)` drives the
real commands against the tiny grid instead of the full snodas spec.
"""

import hashlib

import numpy
import pytest
import rasterio

from click.testing import CliRunner

from snowtool.cli._context import CliContext
from snowtool.snowdb.constants import DEM_HASH_TAG
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.dem_source import LocalFile
from snowtool.snowdb.terrain import (
    ASPECT_COMPONENTS,
    ASPECT_FLAT,
    ASPECT_MAJORITY,
    ELEVATION,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def initialized_root(tmp_path, spec):
    """An initialized snowdb root bound to the synthetic spec."""
    SnowDb.initialize(tmp_path, [spec])
    return tmp_path


@pytest.fixture
def cli_obj(initialized_root, spec, source_dem) -> CliContext:
    """A CliContext over the initialized synthetic snowdb (inject as obj=).

    The default DEM source is a local file so terrain commands never reach 3DEP.
    """
    return CliContext(
        root=initialized_root,
        specs=(spec,),
        dem_source=LocalFile(source_dem),
    )


def _fake_generate_terrain(
    source,
    targets,
    *,
    work_crs=None,
    work_resolution=None,
    force=False,
):
    """Stand in for the streaming engine: write a uniform terrain set per target.

    The real engine reprojects the source to a 10 m grid, which is intractable on
    the degree-scale synthetic CLI grid; these tests only exercise CLI wiring, so
    a uniform write is enough (the engine itself is tested in test_terrain_generate).
    """
    hashes = {}
    for target in targets:
        base = target.grid.base_grid
        shape = (base.rows, base.cols)
        crs = rasterio.crs.CRS.from_wkt(target.grid.crs.to_wkt())
        elevation = numpy.full(shape, 1000.0, dtype='float32')
        dem_hash = hashlib.sha256(elevation.tobytes()).hexdigest()
        tags = {DEM_HASH_TAG: dem_hash}
        target.directory.mkdir(parents=True, exist_ok=True)
        common = {
            'transform': base.transform,
            'crs': crs,
            'tile_size': target.tile_size,
        }
        from snowtool.snowdb.cog import write_cog

        write_cog(
            target.directory / ELEVATION.filename,
            elevation,
            nodata=ELEVATION.nodata,
            tags=tags,
            **common,
        )
        write_cog(
            target.directory / ASPECT_MAJORITY.filename,
            numpy.full(shape, ASPECT_FLAT, dtype='uint8'),
            nodata=ASPECT_MAJORITY.nodata,
            tags=tags,
            **common,
        )
        write_cog(
            target.directory / ASPECT_COMPONENTS.filename,
            numpy.full((2, *shape), numpy.nan, dtype='float32'),
            nodata=ASPECT_COMPONENTS.nodata,
            compute_stats=False,
            tags=tags,
            **common,
        )
        hashes[target.name] = dem_hash
    return hashes


@pytest.fixture(autouse=True)
def _fast_terrain(monkeypatch):
    """Replace the streaming terrain engine with a fast uniform writer for CLI tests."""
    monkeypatch.setattr(
        'snowtool.snowdb.terrain_generate.generate_terrain',
        _fake_generate_terrain,
    )
