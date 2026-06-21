"""Generic, dataset-agnostic snowtool constants.

Dataset-specific values (grid geometry, DEM range, nodata) live on the dataset's
``DatasetSpec`` — see :mod:`snowtool.snowdb.datasets`.
"""

# AOI raster metadata: the grid-tile bounding box the AOI window spans, stored
# as four space-separated ints "ul_row ul_col br_row br_col" (a dataset-agnostic
# tag). The window's upper-left tile is the origin; all tiles in the box are read.
# (Legacy snodas COGs used Bing-quadkey tags; the `snowtool migration aoi-tags`
# command rewrites those to this tag -- see snowtool.migration.aoi_tags.)
TILE_BBOX_TAG = 'SNOWTOOL_TILE_BBOX'

# Feet <-> meters, for elevation-band math.
M_TO_FT = 3.28084
FT_TO_M = 0.3048

# The elevation range (metres) that elevation bands are generated across, shared
# by every dataset. Bands must be comparable across AOIs *and* across datasets,
# so the bracket is a single global constant rather than a per-dataset DEM range:
# the AOI is the geographic unit of interest, not the dataset. The values only
# need to bracket the highest/lowest terrain any AOI can reach -- they are floored
# to whole `band_step_ft` bins in ElevationBand.generate, so a generous bracket
# costs nothing but a few empty (null) bands at the extremes. CONUS spans roughly
# Badwater (-86 m) to Mt. Whitney (4421 m); this brackets it with headroom.
# (Resampling onto the ~1km grids only pulls extremes inward, never outward, so a
# bracket valid for a source DEM can never be exceeded by a resampled one.)
MIN_ELEVATION_M = -100.0
MAX_ELEVATION_M = 4500.0
