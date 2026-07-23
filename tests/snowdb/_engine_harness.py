"""Shared scaffold for the terrain/landcover generation-engine tests.

Both engine test modules bin a tiny synthetic 512 x 512 @ 10 m source in
EPSG:5070 (the terrain work CRS, so reprojection is a near-identity) into a
128 x 128 @ 40 m target -- four fine pixels per target cell. The module
constants, the :func:`make_target` factory, and :func:`run_serial_vs_parallel`
(the hash-equality + array-equality determinism check) are identical across the
two engines and live here; each file keeps its own engine-specific numeric
assertions (west-facing aspect, forest percentages, ...).
"""

import numpy
import rasterio

from snowtool.snowdb.grid import make_grid
from snowtool.snowdb.zones.zone_layer import ZoneLayerTarget

WORK_EPSG = 5070
ORIGIN_X = -500_000.0
ORIGIN_Y = 2_000_000.0
SRC_PX = 10.0
SRC_N = 512
# 128 cells x 40 m == 5120 m == the 512 x 10 m source extent (4 fine px / cell).
TARGET_N = 128
TARGET_PX = 40.0
TARGET_TILE = 128


def make_target(
    directory,
    *,
    name='t',
    px=TARGET_PX,
    n=TARGET_N,
    tile=TARGET_TILE,
):
    """A :class:`ZoneLayerTarget` on the shared 5070 grid rooted at ``directory``."""
    grid = make_grid(
        origin_x=ORIGIN_X,
        origin_y=ORIGIN_Y,
        px_size=px,
        cols=n,
        rows=n,
        tile_size=tile,
        crs=WORK_EPSG,
    )
    return ZoneLayerTarget(
        name=name,
        grid=grid,
        tile_size=tile,
        directory=directory,
    )


def run_serial_vs_parallel(
    engine_fn,
    source_path,
    serial_target,
    parallel_target,
    read_layers,
    *,
    workers,
    block_size,
    hash_key='t',
):
    """Assert a parallel engine pass reproduces the serial pass bit-for-bit.

    Runs ``engine_fn`` once serially (``workers=1``) and once with ``workers``,
    both over the same single ``source_path``, and asserts the generation hash
    (keyed by ``hash_key``) matches and that every layer ``read_layers`` reads
    back is array-equal. Returns the ``(serial_hash, parallel_hash)`` dicts so a
    caller can pin additional per-engine facts.
    """
    with rasterio.open(source_path) as src:
        serial_hash = engine_fn(
            src,
            [serial_target],
            workers=1,
            block_size=block_size,
            force=True,
        )
    with rasterio.open(source_path) as src:
        parallel_hash = engine_fn(
            src,
            [parallel_target],
            workers=workers,
            block_size=block_size,
            force=True,
        )

    assert serial_hash[hash_key] == parallel_hash[hash_key]

    for serial_layer, parallel_layer in zip(
        read_layers(serial_target.directory),
        read_layers(parallel_target.directory),
        strict=True,
    ):
        numpy.testing.assert_array_equal(serial_layer, parallel_layer)

    return serial_hash, parallel_hash
