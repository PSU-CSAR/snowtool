"""The terrain zone-layer definitions: constants, layers, and format version.

The static data the terrain *provider* (:mod:`snowtool.snowdb.zones.terrain`) and its
generation *engine* (:mod:`snowtool.snowdb.zones.terrain_generate`) both need -- the
per-layer nodata sentinels, the aspect-class codes, the :class:`ZoneLayer`
definitions, and the on-disk format version. It lives in its own module so the
engine can import these without importing the provider: the provider imports the
engine to bind its module-level default, so a shared source here is what keeps that
one-directional (provider -> engine -> layers) instead of a cycle.
"""

from __future__ import annotations

from snowtool.snowdb.constants import (
    M_TO_FT,
    MAX_ELEVATION_M,
    MIN_ELEVATION_M,
)
from snowtool.snowdb.zones.zone_layer import ZoneLayer
from snowtool.snowdb.zones.zoning import (
    BandedZoning,
    CategoricalZoning,
    ClassZone,
    EvenBucketZoning,
    ThresholdZoning,
)

# Defaults for the projected, fine work grid aspect is computed on. CONUS Albers
# (metres, near-square) keeps slope/aspect undistorted; 10 m matches 3DEP. These
# are only fallbacks -- the DemSource supplies the right values for its data (see
# the terrain_generate module docstring), since the work resolution must track the
# source's native resolution and the work CRS its region. They live here (not in
# terrain_generate) so terrain_source can read them without importing the engine --
# the engine imports the source's DemSource type, so the reverse would cycle.
DEFAULT_WORK_CRS = 'EPSG:5070'
DEFAULT_WORK_RESOLUTION = 10.0

# On-disk format version of a terrain layer set, owned by the terrain provider and
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
    ),
)


def _component_zoning() -> EvenBucketZoning:
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
