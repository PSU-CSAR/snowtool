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
