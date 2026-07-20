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

from snowtool.snowdb.constants import (
    DEM_HASH_TAG,
    M_TO_FT,
    MAX_ELEVATION_M,
    MIN_ELEVATION_M,
)
from snowtool.snowdb.progress import NULL_PROGRESS, ProgressReporter
from snowtool.snowdb.zones.zone_layer import (
    GenerationOptions,
    ZoneLayer,
    ZoneLayerProvider,
)
from snowtool.snowdb.zones.zoning import (
    BandedZoning,
    CategoricalZoning,
    ClassZone,
    EvenBucketZoning,
    ThresholdZoning,
)

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


# On-disk format version of a terrain layer set, owned by TerrainProvider and
# stamped (via provenance.versioned_hash) onto DEM_HASH_TAG by the generator. Bump
# on a material change to the terrain layer encoding so existing sets read as stale.
# v2 added aspect_entropy.tif. v3 split the single two-band aspect_components.tif
# into the single-band northness.tif + eastness.tif query axes (finite nodata).
TERRAIN_FORMAT_VERSION = 3

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
# Northness/eastness are means of cos/sin(aspect) in [-1, 1]. A far-below-domain
# finite sentinel (NOT NaN) marks a cell with no non-flat pixels, so it digitises
# cleanly out of the banded scheme (a NaN would compare out of nothing, so the
# BandedZoning.assign nodata mask could not exclude it -- the same reason elevation
# uses a finite sentinel).
ASPECT_COMPONENT_NODATA = -9999.0
# Entropy is normalised into [0, 1], so a far-out sentinel doubles as nodata and
# digitises cleanly out of the threshold split (unlike NaN, which == nothing and so
# could not be excluded by ThresholdZoning.assign).
ASPECT_ENTROPY_NODATA = -1.0

# Default split for the aspect-direction entropy zone: a cell below this reads as a
# coherent, high-signal aspect, at/above it as a mixed, low-signal one. Overridable
# per query (the dataset zones param ``entropy_threshold``, or a ``:override`` token).
DEFAULT_ASPECT_ENTROPY_THRESHOLD = 0.5

# Northness/eastness are the mean cos/sin of aspect: dimensionless in [-1, 1], where
# a band *width* carries no external meaning. So they are bucketed, not stepped -- the
# closed domain is cut into a user-adjustable count of equal buckets (the dataset zones
# param ``buckets``, or a query ``:override``), default 4 ([-1,-0.5), [-0.5,0), [0,0.5),
# [0.5,1]). No fabricated unit, no arbitrary width.
ASPECT_COMPONENT_DOMAIN_MIN = -1
ASPECT_COMPONENT_DOMAIN_MAX = 1
DEFAULT_ASPECT_COMPONENT_BUCKETS = 4


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
    zoning=BandedZoning(
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
    zoning=CategoricalZoning(
        classes=(
            ClassZone(key='N', label='N', code=ASPECT_N),
            ClassZone(key='E', label='E', code=ASPECT_E),
            ClassZone(key='S', label='S', code=ASPECT_S),
            ClassZone(key='W', label='W', code=ASPECT_W),
            ClassZone(key='flat', label='flat', code=ASPECT_FLAT),
        ),
        layer_nodata=ASPECT_MAJORITY_NODATA,
    ),
)


def _component_zoning():
    """The shared bucketed scheme for a northness/eastness axis over ``[-1, 1]``.

    Both components have the identical domain and bucket count, so they share one
    scheme factory (each axis gets its own instance, keyed by its own layer key).
    """
    return EvenBucketZoning(
        domain_min=ASPECT_COMPONENT_DOMAIN_MIN,
        domain_max=ASPECT_COMPONENT_DOMAIN_MAX,
        default_buckets=DEFAULT_ASPECT_COMPONENT_BUCKETS,
        layer_nodata=ASPECT_COMPONENT_NODATA,
    )


NORTHNESS = ZoneLayer(
    filename='northness.tif',
    dtype='float32',
    nodata=ASPECT_COMPONENT_NODATA,
    band_descriptions=('northness_mean_cos_aspect',),
    key='northness',
    # Bucketed mean cos(aspect): +1 due north, -1 due south.
    zoning=_component_zoning(),
)
EASTNESS = ZoneLayer(
    filename='eastness.tif',
    dtype='float32',
    nodata=ASPECT_COMPONENT_NODATA,
    band_descriptions=('eastness_mean_sin_aspect',),
    key='eastness',
    # Bucketed mean sin(aspect): +1 due east, -1 due west.
    zoning=_component_zoning(),
)
ASPECT_ENTROPY = ZoneLayer(
    filename='aspect_entropy.tif',
    dtype='float32',
    nodata=ASPECT_ENTROPY_NODATA,
    band_descriptions=('aspect_dir_entropy_norm_5class',),
    key='aspect_entropy',
    # Normalised Shannon entropy over the five aspect classes (the counts that pick
    # the majority): low = coherent (high signal), high = mixed (low signal). A
    # below/at-or-above split, so a query keeps only well-supported aspect cells.
    # Meant to be crossed with the majority axis (terrain.aspect): a flat-dominated
    # cell is low-entropy *and* majority flat, so it never poses as a high-signal
    # direction -- the flat case is owned by the majority class, not the entropy.
    zoning=ThresholdZoning(
        default_threshold=DEFAULT_ASPECT_ENTROPY_THRESHOLD,
        domain_min=0,
        domain_max=1,
        # Normalised Shannon entropy H in [0, 1]; 'Hnorm' names the quantity (so the
        # 0.5 split reads as "half of maximum directional entropy") rather than the
        # vaguer 'frac'.
        unit='Hnorm',
        value_scale=1,
        layer_nodata=ASPECT_ENTROPY_NODATA,
        below_label='high_signal',
        above_label='low_signal',
        param_key='entropy_threshold',
    ),
)

# Every layer of a complete terrain set, in write order.
TERRAIN_LAYERS = (ELEVATION, ASPECT_MAJORITY, NORTHNESS, EASTNESS, ASPECT_ENTROPY)


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

        engine = self._engine
        if engine is None:
            from snowtool.snowdb.zones.terrain_generate import generate_terrain

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
                progress=progress,
            )
