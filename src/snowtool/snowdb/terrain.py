"""The *terrain* zone-layer provider: elevation + aspect layers from a DEM.

Replaces the single ``dem.tif``. Because aspect cannot be resampled after the
fact (it must be computed from elevation at the source resolution), terrain is
generated once from a fine-resolution DEM (see
:mod:`snowtool.snowdb.terrain_generate`) into a small family of co-registered
layers on the dataset grid, stored under ``data/<name>/terrain/``:

* ``elevation.tif`` -- ``float32`` mean elevation (m).
* ``aspect_majority.tif`` -- ``uint8`` majority aspect class per cell
  (``0`` N, ``1`` E, ``2`` S, ``3`` W, ``4`` flat); nodata ``255``.
* ``aspect_components.tif`` -- two ``float32`` bands, ``northness`` =
  mean ``cos(aspect)`` and ``eastness`` = mean ``sin(aspect)`` over the cell's
  non-flat pixels (the first circular moment; ``hypot(northness, eastness)`` is
  the orientation purity in ``[0, 1]``).

Every layer carries a :data:`~snowtool.snowdb.constants.DEM_HASH_TAG` tag -- the
sha256 of the generated elevation array -- so the whole set's provenance can be
read back cheaply.

:class:`TerrainProvider` is the :class:`~snowtool.snowdb.zone_layer.ZoneLayerProvider`
for this kind: it names the layers, the ``terrain/`` subdirectory, and the DEM
source/engine, so a dataset builds and reads terrain like any other zone layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from snowtool.snowdb.constants import (
    DEM_HASH_TAG,
    M_TO_FT,
    MAX_ELEVATION_M,
    MIN_ELEVATION_M,
)
from snowtool.snowdb.zone_layer import GenerationOptions, ZoneLayer, ZoneLayerProvider
from snowtool.snowdb.zoning import ClassZone, banded, categorical

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from snowtool.snowdb.zone_layer import Bounds, ZoneLayerSource, ZoneLayerTarget

    # The terrain generation engine (terrain_generate.generate_terrain); injectable
    # so a test can supply a fast stand-in. Returns the per-target provenance hash.
    TerrainEngine = Callable[..., dict[str, str]]

# On-disk format version of a terrain layer set, owned by TerrainProvider and
# stamped (via provenance.versioned_hash) onto DEM_HASH_TAG by the generator. Bump
# on a material change to the terrain layer encoding so existing sets read as stale.
TERRAIN_FORMAT_VERSION = 1

# Aspect majority classes (cell's modal cardinal quadrant, or flat).
ASPECT_N = 0
ASPECT_E = 1
ASPECT_S = 2
ASPECT_W = 3
ASPECT_FLAT = 4

# Per-layer nodata sentinels. Elevation uses a far-below-bracket sentinel rather
# than NaN so it digitizes cleanly out of every elevation band (a NaN would not
# compare out), which is exactly how query-time banding excludes uncovered cells.
ELEVATION_NODATA = -9999.0
ASPECT_MAJORITY_NODATA = 255
ASPECT_COMPONENTS_NODATA = float('nan')


ELEVATION = ZoneLayer(
    filename='elevation.tif',
    dtype='float32',
    nodata=ELEVATION_NODATA,
    band_descriptions=('elevation_mean_m',),
    key='elevation',
    # Elevation bands span the global bracket in feet, aligned to 0, so a band
    # means the same thing across AOIs and datasets; pixels are metres, so the
    # scheme scales by M_TO_FT. The per-dataset default step is spec.band_step_ft
    # (passed at query time); default_step here is the fallback.
    zoning=banded(
        domain_min=MIN_ELEVATION_M * M_TO_FT,
        domain_max=MAX_ELEVATION_M * M_TO_FT,
        default_step=1000,
        unit='ft',
        value_scale=M_TO_FT,
        layer_nodata=ELEVATION_NODATA,
    ),
)
ASPECT_MAJORITY = ZoneLayer(
    filename='aspect_majority.tif',
    dtype='uint8',
    nodata=ASPECT_MAJORITY_NODATA,
    band_descriptions=('majority_cls_0N1E2S3W4flat',),
    key='aspect',
    zoning=categorical(
        (
            ClassZone(key='N', label='N', code=ASPECT_N),
            ClassZone(key='E', label='E', code=ASPECT_E),
            ClassZone(key='S', label='S', code=ASPECT_S),
            ClassZone(key='W', label='W', code=ASPECT_W),
            ClassZone(key='flat', label='flat', code=ASPECT_FLAT),
        ),
        layer_nodata=ASPECT_MAJORITY_NODATA,
    ),
)
ASPECT_COMPONENTS = ZoneLayer(
    filename='aspect_components.tif',
    dtype='float32',
    nodata=ASPECT_COMPONENTS_NODATA,
    band_descriptions=('northness_mean_cos_aspect', 'eastness_mean_sin_aspect'),
    key='aspect_components',
)

# Every layer of a complete terrain set, in write order.
TERRAIN_LAYERS = (ELEVATION, ASPECT_MAJORITY, ASPECT_COMPONENTS)


class TerrainProvider(ZoneLayerProvider):
    """The terrain zone-layer kind: elevation + aspect, derived from a DEM."""

    name = 'terrain'
    subdir = 'terrain'
    layers = TERRAIN_LAYERS
    hash_tag = DEM_HASH_TAG
    format_version = TERRAIN_FORMAT_VERSION

    def __init__(self: Self, engine: TerrainEngine | None = None) -> None:
        # The generation engine, injectable for tests; None resolves lazily to the
        # real streaming engine in generate() (terrain_generate imports terrain, so
        # a module-level import would cycle).
        self._engine = engine

    def default_source(self: Self, root: Path) -> ZoneLayerSource:
        """The default DEM source -- USGS 3DEP streamed from the public bucket."""
        from snowtool.snowdb.dem_source import ThreeDEP

        return ThreeDEP()

    def local_source(self: Self, path: Path) -> ZoneLayerSource:
        """A local on-disk DEM file source (the ``--source terrain PATH`` path)."""
        from snowtool.snowdb.dem_source import LocalFile

        return LocalFile(path)

    def generate(
        self: Self,
        source: ZoneLayerSource,
        targets: list[ZoneLayerTarget],
        bounds: Bounds,
        *,
        force: bool = False,
        options: GenerationOptions | None = None,
    ) -> dict[str, str]:
        """Stream the DEM ``source`` once, binning terrain into every target.

        ``options`` carries the engine's block-level parallelism knobs
        (``workers``, ``block_size``); the DEM source supplies the projected work
        grid (``work_crs``/``work_resolution``).
        """
        from snowtool.snowdb.dem_source import DemSource

        engine = self._engine
        if engine is None:
            from snowtool.snowdb.terrain_generate import generate_terrain

            engine = generate_terrain

        if not isinstance(source, DemSource):  # pragma: no cover - defensive
            raise TypeError(f'terrain generation needs a DemSource, got {source!r}')
        options = options or GenerationOptions()
        with source.open(bounds) as src:
            return engine(
                src,
                targets,
                work_crs=source.work_crs,
                work_resolution=source.work_resolution,
                workers=options.workers,
                block_size=options.block_size,
                force=force,
            )
