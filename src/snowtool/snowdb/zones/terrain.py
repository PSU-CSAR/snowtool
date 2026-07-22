"""The *terrain* zone-layer provider: elevation + aspect layers from a DEM.

Because aspect cannot be resampled after the fact (it must be computed from
elevation at the source resolution), terrain is generated once from a fine-resolution
DEM (see :mod:`snowtool.snowdb.zones.terrain_generate`) into a small family of
co-registered layers on the dataset grid, stored under ``data/<name>/terrain/``:

* ``elevation.tif`` -- ``float32`` mean elevation (m).
* ``aspect_majority.tif`` -- ``uint8`` majority aspect class per cell
  (``0`` N, ``1`` E, ``2`` S, ``3`` W, ``4`` flat); nodata ``255``.
* ``northness.tif`` / ``eastness.tif`` -- two ``float32`` single-band layers,
  ``northness`` = mean ``cos(aspect)`` and ``eastness`` = mean ``sin(aspect)``
  over the cell's non-flat pixels (the first circular moment; ``hypot(northness,
  eastness)`` is the orientation purity in ``[0, 1]``). Each is its own query-able
  zone axis, banded over ``[-1, 1]`` (see :data:`NORTHNESS`/:data:`EASTNESS`); a
  cell with no non-flat pixels carries the finite :data:`ASPECT_COMPONENT_NODATA`
  sentinel. Two single-band files rather than one two-band file because the
  :class:`~snowtool.snowdb.zones.zone_layer.ZoneLayer` model is one file + one
  band + one zoning scheme per query key, and the tiled reader reads band 0 only.
* ``aspect_entropy.tif`` -- ``float32`` normalised Shannon entropy of the cell's
  aspect-class distribution (the same five N/E/S/W/flat counts that feed the
  majority vote), in ``[0, 1]``: ``0`` = every pixel one class (coherent),
  ``1`` = evenly mixed; nodata ``-1``. Thresholded into a high-/low-signal zone
  so a query can keep only cells whose majority aspect is well-supported.

Every layer carries a :data:`~snowtool.snowdb.constants.DEM_HASH_TAG` tag -- the
sha256 of the generated elevation array -- so the whole set's provenance can be
read back cheaply.

:class:`TerrainProvider` is the
:class:`~snowtool.snowdb.zones.zone_layer.ZoneLayerProvider` for this kind: it names
the layers, the ``terrain/`` subdirectory, and the DEM source/engine, so a dataset
builds and reads terrain like any other zone layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self

from snowtool.snowdb.constants import DEM_HASH_TAG
from snowtool.snowdb.progress import NULL_PROGRESS, ProgressReporter
from snowtool.snowdb.zones.terrain_generate import generate_terrain
from snowtool.snowdb.zones.terrain_layers import (
    ASPECT_COMPONENT_DOMAIN_MAX,
    ASPECT_COMPONENT_DOMAIN_MIN,
    ASPECT_COMPONENT_NODATA,
    ASPECT_E,
    ASPECT_ENTROPY,
    ASPECT_ENTROPY_NODATA,
    ASPECT_FLAT,
    ASPECT_MAJORITY,
    ASPECT_MAJORITY_NODATA,
    ASPECT_N,
    ASPECT_S,
    ASPECT_W,
    DEFAULT_ASPECT_COMPONENT_BUCKETS,
    DEFAULT_ASPECT_ENTROPY_THRESHOLD,
    EASTNESS,
    ELEVATION,
    ELEVATION_NODATA,
    NORTHNESS,
    TERRAIN_FORMAT_VERSION,
    TERRAIN_LAYERS,
)
from snowtool.snowdb.zones.zone_layer import GenerationOptions, ZoneLayerProvider

# The layer/constant definitions live in ``terrain_layers`` so the generation
# engine can import them without importing this provider (this module imports the
# engine below to bind its module-level default). Re-exported here so external
# importers keep reading them off ``snowtool.snowdb.zones.terrain``.
__all__ = [
    'ASPECT_COMPONENT_DOMAIN_MAX',
    'ASPECT_COMPONENT_DOMAIN_MIN',
    'ASPECT_COMPONENT_NODATA',
    'ASPECT_E',
    'ASPECT_ENTROPY',
    'ASPECT_ENTROPY_NODATA',
    'ASPECT_FLAT',
    'ASPECT_MAJORITY',
    'ASPECT_MAJORITY_NODATA',
    'ASPECT_N',
    'ASPECT_S',
    'ASPECT_W',
    'DEFAULT_ASPECT_COMPONENT_BUCKETS',
    'DEFAULT_ASPECT_ENTROPY_THRESHOLD',
    'EASTNESS',
    'ELEVATION',
    'ELEVATION_NODATA',
    'NORTHNESS',
    'TERRAIN_FORMAT_VERSION',
    'TERRAIN_LAYERS',
    'TerrainProvider',
]

if TYPE_CHECKING:
    from pathlib import Path

    import rasterio.io

    from snowtool.snowdb.grid import Bounds
    from snowtool.snowdb.zones.zone_layer import ZoneLayerSource, ZoneLayerTarget

    class TerrainEngine(Protocol):
        """The terrain generation engine's call signature.

        Matches :func:`~snowtool.snowdb.zones.terrain_generate.generate_terrain`;
        injectable (see :meth:`TerrainProvider.__init__`) so a test can supply a
        fast stand-in that is signature-checked against the real engine. Returns
        the per-target provenance hash (all values equal).
        """

        def __call__(
            self,
            source: rasterio.io.DatasetReader,
            targets: list[ZoneLayerTarget],
            *,
            work_crs: str = ...,
            work_resolution: float | None = ...,
            workers: int | None = ...,
            block_size: int | None = ...,
            force: bool = ...,
            progress: ProgressReporter = ...,
        ) -> dict[str, str]: ...


class TerrainProvider(ZoneLayerProvider):
    """The terrain zone-layer kind: elevation + aspect, derived from a DEM."""

    name = 'terrain'
    subdir = 'terrain'
    layers = TERRAIN_LAYERS
    hash_tag = DEM_HASH_TAG
    format_version = TERRAIN_FORMAT_VERSION

    def __init__(self: Self, engine: TerrainEngine | None = None) -> None:
        # The generation engine, injectable for tests; None binds the real
        # streaming engine (terrain_generate imports terrain_layers, not this
        # provider, so the default can be a plain module-level import).
        self._engine: TerrainEngine = engine if engine is not None else generate_terrain

    def default_source(self: Self, root: Path) -> ZoneLayerSource:
        """The default DEM source -- USGS 3DEP streamed from the public bucket."""
        from snowtool.snowdb.zones.terrain_source import ThreeDEP

        return ThreeDEP()

    def local_source(self: Self, path: Path) -> ZoneLayerSource:
        """A local on-disk DEM file source (the ``--source terrain PATH`` path)."""
        from snowtool.snowdb.zones.terrain_source import LocalFile

        return LocalFile(path)

    def generate(
        self: Self,
        source: ZoneLayerSource,
        targets: list[ZoneLayerTarget],
        bounds: Bounds,
        *,
        force: bool = False,
        options: GenerationOptions | None = None,
        progress: ProgressReporter = NULL_PROGRESS,
    ) -> dict[str, str]:
        """Stream the DEM ``source`` once, binning terrain into every target.

        ``options`` carries the engine's block-level parallelism knobs
        (``workers``, ``block_size``); the DEM source supplies the projected work
        grid (``work_crs``/``work_resolution``). ``progress`` reports the engine's
        per-block reprojection.
        """
        from snowtool.snowdb.zones.terrain_source import DemSource

        if not isinstance(source, DemSource):  # pragma: no cover - defensive
            raise TypeError(f'terrain generation needs a DemSource, got {source!r}')
        options = options or GenerationOptions()
        with source.open(bounds) as src:
            return self._engine(
                src,
                targets,
                work_crs=source.work_crs,
                work_resolution=source.work_resolution,
                workers=options.workers,
                block_size=options.block_size,
                force=force,
                progress=progress,
            )
