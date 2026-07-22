"""The land-cover zone-layer definitions: the forest-cover layer + format version.

The static data the land-cover *provider*
(:mod:`snowtool.snowdb.zones.landcover`) and its generation *engine*
(:mod:`snowtool.snowdb.zones.landcover_generate`) both need -- the
percent-forest-cover :class:`ZoneLayer` and the on-disk format version. It lives in
its own module so the engine can import these without importing the provider: the
provider imports the engine to bind its module-level default, so a shared source
here is what keeps that one-directional (provider -> engine -> layers) instead of a
cycle. The provider re-exports every name here, so external importers still read
them off ``snowtool.snowdb.zones.landcover``.
"""

from __future__ import annotations

from snowtool.snowdb.constants import FOREST_PCT_NODATA
from snowtool.snowdb.zones.zone_layer import ZoneLayer
from snowtool.snowdb.zones.zoning import ThresholdZoning

# On-disk format version of a land-cover layer set, owned by LandCoverProvider and
# stamped (via provenance.versioned_hash) onto NLCD_HASH_TAG by the generator. Bump
# on a material change to the land-cover layer encoding so existing sets read stale.
LANDCOVER_FORMAT_VERSION = 1

# Default forest-cover threshold (percent): cells with this much forest or more
# read as "forested", below it as "unforested". Overridable per query.
DEFAULT_FOREST_THRESHOLD_PCT = 50


FOREST_COVER = ZoneLayer(
    filename='forest_cover_pct.tif',
    dtype='uint8',
    nodata=FOREST_PCT_NODATA,
    band_descriptions=('forest_cover_percent_0_100',),
    key='forest_cover',
    # Forest cover is a forested/unforested split at a percent threshold (default
    # 50%), not a set of percent bands: the question is whether a cell is forested,
    # and the threshold is the per-query knob. Pixels are already percent, so
    # value_scale is 1.
    zoning=ThresholdZoning(
        default_threshold=DEFAULT_FOREST_THRESHOLD_PCT,
        domain_min=0,
        domain_max=100,
        unit='%',
        value_scale=1,
        layer_nodata=FOREST_PCT_NODATA,
        below_label='unforested',
        above_label='forested',
    ),
)

# Every layer of a complete land-cover set, in write order.
LANDCOVER_LAYERS = (FOREST_COVER,)
