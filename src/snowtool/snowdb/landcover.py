"""The *land-cover* zone-layer provider: the NLCD percent-forest-cover layer.

Parallels :mod:`snowtool.snowdb.terrain` but comes from a different source (NLCD
land cover, not a DEM) and so carries its own provenance
(:data:`~snowtool.snowdb.constants.NLCD_HASH_TAG`, not the DEM hash). It is
generated once from a fine-resolution NLCD raster (see
:mod:`snowtool.snowdb.landcover_generate`) onto the dataset grid, stored under
``data/<name>/landcover/``:

* ``forest_cover_pct.tif`` -- ``uint8`` percent forest cover (0..100), the share
  of the cell's NLCD pixels classed as forest (see
  :data:`~snowtool.snowdb.constants.FOREST_CLASSES`); nodata ``255``.

:class:`LandCoverProvider` is the
:class:`~snowtool.snowdb.zone_layer.ZoneLayerProvider` for this kind, so a dataset
builds and reads land cover like any other zone layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from snowtool.snowdb.constants import FOREST_PCT_NODATA, NLCD_HASH_TAG
from snowtool.snowdb.zone_layer import ZoneLayer, ZoneLayerProvider
from snowtool.snowdb.zoning import threshold

# On-disk format version of a land-cover layer set, owned by LandCoverProvider and
# stamped (via provenance.versioned_hash) onto NLCD_HASH_TAG by the generator. Bump
# on a material change to the land-cover layer encoding so existing sets read stale.
LANDCOVER_FORMAT_VERSION = 1

# Default forest-cover threshold (percent): cells with this much forest or more
# read as "forested", below it as "unforested". Overridable per query.
DEFAULT_FOREST_THRESHOLD_PCT = 50

if TYPE_CHECKING:
    from pathlib import Path

    from snowtool.snowdb.zone_layer import Bounds, ZoneLayerSource, ZoneLayerTarget


FOREST_COVER = ZoneLayer(
    filename='forest_cover_pct.tif',
    dtype='uint8',
    nodata=FOREST_PCT_NODATA,
    band_descriptions=('forest_cover_percent_0_100',),
    key='forest_cover',
    # Forest cover is a forested/unforested split at a percent threshold (default
    # 40%), not a set of percent bands: the question is whether a cell is forested,
    # and the threshold is the per-query knob. Pixels are already percent, so
    # value_scale is 1.
    zoning=threshold(
        default_threshold=DEFAULT_FOREST_THRESHOLD_PCT,
        unit='%',
        value_scale=1,
        layer_nodata=FOREST_PCT_NODATA,
        below_label='unforested',
        above_label='forested',
    ),
)

# Every layer of a complete land-cover set, in write order.
LANDCOVER_LAYERS = (FOREST_COVER,)


class LandCoverProvider(ZoneLayerProvider):
    """The land-cover zone-layer kind: percent forest cover, derived from NLCD."""

    name = 'landcover'
    subdir = 'landcover'
    layers = LANDCOVER_LAYERS
    hash_tag = NLCD_HASH_TAG
    format_version = LANDCOVER_FORMAT_VERSION

    def default_source(self: Self, root: Path) -> ZoneLayerSource:
        """The default NLCD source -- the MRLC Annual NLCD bundle, cached locally.

        Cached under the snowdb ``root`` so a repeated init reuses the (large)
        download.
        """
        from snowtool.snowdb.landcover_source import AnnualNLCD

        return AnnualNLCD(cache_dir=root / '.cache' / 'landcover')

    def local_source(self: Self, path: Path) -> ZoneLayerSource:
        """A local on-disk NLCD file source (the ``--source landcover PATH`` path)."""
        from snowtool.snowdb.landcover_source import LocalFile

        return LocalFile(path)

    def generate(
        self: Self,
        source: ZoneLayerSource,
        targets: list[ZoneLayerTarget],
        bounds: Bounds,
        *,
        force: bool = False,
        **options: object,
    ) -> dict[str, str]:
        """Stream the NLCD ``source`` once, binning forest cover into every target."""
        from snowtool.snowdb.landcover_generate import generate_landcover

        with source.open(bounds) as src:
            return generate_landcover(src, targets, force=force)
