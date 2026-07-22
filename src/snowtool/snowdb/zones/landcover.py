"""The *land-cover* zone-layer provider: the NLCD percent-forest-cover layer.

Parallels :mod:`snowtool.snowdb.zones.terrain` but comes from a different source (NLCD
land cover, not a DEM) and so carries its own provenance
(:data:`~snowtool.snowdb.constants.NLCD_HASH_TAG`, not the DEM hash). It is
generated once from a fine-resolution NLCD raster (see
:mod:`snowtool.snowdb.zones.landcover_generate`) onto the dataset grid, stored under
``data/<name>/landcover/``:

* ``forest_cover_pct.tif`` -- ``uint8`` percent forest cover (0..100), the share
  of the cell's NLCD pixels classed as forest (see
  :data:`~snowtool.snowdb.constants.FOREST_CLASSES`); nodata ``255``.

:class:`LandCoverProvider` is the
:class:`~snowtool.snowdb.zones.zone_layer.ZoneLayerProvider` for this kind, so a dataset
builds and reads land cover like any other zone layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self

from snowtool.snowdb.constants import NLCD_HASH_TAG
from snowtool.snowdb.progress import NULL_PROGRESS, ProgressReporter
from snowtool.snowdb.zones.landcover_generate import generate_landcover
from snowtool.snowdb.zones.landcover_layers import (
    DEFAULT_FOREST_THRESHOLD_PCT,
    FOREST_COVER,
    LANDCOVER_FORMAT_VERSION,
    LANDCOVER_LAYERS,
)
from snowtool.snowdb.zones.zone_layer import ZoneLayerProvider

# The layer/constant definitions live in ``landcover_layers`` so the generation
# engine can import them without importing this provider (this module imports the
# engine below to bind its module-level default). Re-exported here so external
# importers keep reading them off ``snowtool.snowdb.zones.landcover``.
__all__ = [
    'DEFAULT_FOREST_THRESHOLD_PCT',
    'FOREST_COVER',
    'LANDCOVER_FORMAT_VERSION',
    'LANDCOVER_LAYERS',
    'LandCoverProvider',
]

if TYPE_CHECKING:
    from pathlib import Path

    import rasterio.io

    from snowtool.snowdb.grid import Bounds
    from snowtool.snowdb.zones.zone_layer import (
        GenerationOptions,
        ZoneLayerSource,
        ZoneLayerTarget,
    )

    class LandCoverEngine(Protocol):
        """The land-cover generation engine's call signature.

        Matches
        :func:`~snowtool.snowdb.zones.landcover_generate.generate_landcover`;
        injectable (see :meth:`LandCoverProvider.__init__`) so a test can supply a
        fast stand-in that is signature-checked against the real engine. Returns
        the per-target provenance hash (all values equal).
        """

        def __call__(
            self,
            source: rasterio.io.DatasetReader,
            targets: list[ZoneLayerTarget],
            *,
            workers: int | None = ...,
            block_size: int | None = ...,
            force: bool = ...,
            progress: ProgressReporter = ...,
        ) -> dict[str, str]: ...


class LandCoverProvider(ZoneLayerProvider):
    """The land-cover zone-layer kind: percent forest cover, derived from NLCD."""

    name = 'landcover'
    subdir = 'landcover'
    layers = LANDCOVER_LAYERS
    hash_tag = NLCD_HASH_TAG
    format_version = LANDCOVER_FORMAT_VERSION

    def __init__(self: Self, engine: LandCoverEngine | None = None) -> None:
        # The generation engine, injectable for tests; None binds the real
        # streaming engine (landcover_generate imports landcover_layers, not this
        # provider, so the default can be a plain module-level import).
        self._engine: LandCoverEngine = (
            engine if engine is not None else generate_landcover
        )

    def default_source(self: Self, root: Path) -> ZoneLayerSource:
        """The default NLCD source -- the MRLC Annual NLCD bundle, cached locally.

        Cached under the snowdb ``root`` so a repeated init reuses the (large)
        download.
        """
        from snowtool.snowdb.zones.landcover_source import AnnualNLCD

        return AnnualNLCD(cache_dir=root / '.cache' / 'landcover')

    def local_source(self: Self, path: Path) -> ZoneLayerSource:
        """A local on-disk NLCD file source (the ``--source landcover PATH`` path)."""
        from snowtool.snowdb.zones.landcover_source import LocalFile

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
        """Stream the NLCD ``source`` once, binning forest cover into every target.

        ``options`` carries the engine's block-level parallelism knobs
        (``workers``, ``block_size``). ``progress`` reports the source's download
        (the heavy step for the default ~1.5 GB Annual NLCD source) and then the
        engine's per-block binning.
        """
        from snowtool.snowdb.zones.landcover_source import LandCoverSource
        from snowtool.snowdb.zones.zone_layer import GenerationOptions

        if not isinstance(source, LandCoverSource):  # pragma: no cover - defensive
            raise TypeError(
                f'land-cover generation needs a LandCoverSource, got {source!r}',
            )
        options = options or GenerationOptions()
        with source.open(bounds, progress=progress) as src:
            return self._engine(
                src,
                targets,
                workers=options.workers,
                block_size=options.block_size,
                force=force,
                progress=progress,
            )
