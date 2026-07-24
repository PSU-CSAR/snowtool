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

from typing import TYPE_CHECKING, Self

from snowtool.snowdb.constants import NLCD_HASH_TAG
from snowtool.snowdb.zones.landcover_generate import generate_landcover
from snowtool.snowdb.zones.landcover_layers import (
    LANDCOVER_FORMAT_VERSION,
    LANDCOVER_LAYERS,
)
from snowtool.snowdb.zones.zone_layer import ZoneLayerProvider

if TYPE_CHECKING:
    from pathlib import Path

    from snowtool.snowdb.zones.zone_layer import ZoneLayerSource


class LandCoverProvider(ZoneLayerProvider):
    """The land-cover zone-layer kind: percent forest cover, derived from NLCD.

    The layer/format-version definitions live in ``landcover_layers`` (so the
    engine can import them without importing this provider); the provider only
    wires them up plus the NLCD source and default engine.
    """

    name = 'landcover'
    subdir = 'landcover'
    layers = LANDCOVER_LAYERS
    hash_tag = NLCD_HASH_TAG
    format_version = LANDCOVER_FORMAT_VERSION
    _default_engine = staticmethod(generate_landcover)

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
