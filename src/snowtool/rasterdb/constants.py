# new SNODAS grid values
ORIGIN_X = -124.733333333333333
ORIGIN_Y = 52.875000000000000
ANTIORIGIN_X = -66.9416666666666667
ANTIORIGIN_Y = 24.9500000000000000
PX_SIZE = 0.008333333333333
COLS = 6935
ROWS = 3351
TILE_SIZE = 256
TILE_NATIVE_ZOOM = 4
NODATA = -9999

# AOI raster metadata: the grid-tile bounding box the AOI window spans, stored
# as four space-separated ints "ul_row ul_col br_row br_col" (a dataset-agnostic
# tag). The window's upper-left tile is the origin; all tiles in the box are read.
TILE_BBOX_TAG = 'SNOWTOOL_TILE_BBOX'

# Legacy (snodas) AOI metadata: an origin tile + per-tile intersected set, each
# tag value a Bing quadkey. Read-only, for backwards compatibility with
# already-written COGs.
LEGACY_ORIGIN_TILE_TAG = 'SNODAS_ORIGIN_TILE'
LEGACY_TILE_TAG_PREFIX = 'SNODAS_TILE'

# we use these overall min/max elevation values
# to establish our elevation band ranges
DEM_MIN_M = -84.833877563477
DEM_MAX_M = 4291.7211914062
M_TO_FT = 3.28084
FT_TO_M = 0.3048
