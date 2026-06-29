"""Generic, dataset-agnostic snowtool constants.

Dataset-specific values (grid geometry, DEM range, nodata) live on the dataset's
``DatasetSpec`` — see :mod:`snowtool.snowdb.datasets`. The on-disk *format
version* each provenance tag carries is owned by that artifact's producer (a
zone-layer provider's ``format_version``; the AOI writer's
``AOI_RASTER_FORMAT_VERSION`` in :mod:`snowtool.snowdb.dataset`), and the
``versioned_hash``/``parse_format_version`` helpers live in
:mod:`snowtool.snowdb.provenance`.
"""

# AOI raster metadata: the grid-tile bounding box the AOI window spans, stored
# as four space-separated ints "ul_row ul_col br_row br_col" (a dataset-agnostic
# tag). The window's upper-left tile is the origin; all tiles in the box are read.
# (Legacy snodas COGs used Bing-quadkey tags; the `snowtool migration aoi-tags`
# command rewrites those to this tag -- see snowtool.migration.aoi_tags.)
TILE_BBOX_TAG = 'SNOWTOOL_TILE_BBOX'

# AOI rasters burn per-pixel cell area (m^2) inside the basin polygon and 0
# (= nodata) outside -- so the raster is both the in/out membership signal and the
# area weights, with no separate area raster. Elevation/aspect are read live from
# the terrain set at query time, so the AOI raster carries no DEM-derived values.
AOI_MASK_NODATA = 0

# AOI raster provenance tag: a ``versioned_hash`` of the AOI basin polygon's WKB
# sha256 (see Pourpoint.geometry_hash), carrying the AOI writer's
# AOI_RASTER_FORMAT_VERSION. An AOI raster is stale when this tag no longer matches
# the AOI's current versioned hash -- a changed basin OR a format bump -- a cheap
# tag-only read that drives `aoi rasterize`'s missing-or-stale rebuild without
# opening the full raster.
AOI_HASH_TAG = 'SNOWTOOL_AOI_HASH'

# Terrain provenance tag: a ``versioned_hash`` of the generated mean-elevation
# array's sha256 (carrying TerrainProvider.format_version), stamped on every layer
# of a dataset's terrain set (elevation + aspect). It identifies the DEM the whole
# set was derived from, so a terrain set can be reconciled against the source it
# came from. Unlike the AOI hash, this never rides on AOI rasters: AOI rasters are
# decoupled from the DEM (they carry cell area, not elevation), so elevation/aspect
# are read live from the terrain set at query time and a terrain rebuild needs no
# AOI rebuild.
DEM_HASH_TAG = 'SNOWTOOL_DEM_HASH'

# Land-cover provenance tag: a ``versioned_hash`` of the generated percent-forest
# array's sha256 (carrying LandCoverProvider.format_version), stamped on every
# layer of a dataset's land-cover set. It identifies the NLCD source the layer was
# derived from -- the land-cover analogue of DEM_HASH_TAG. Like the DEM hash (and
# unlike the AOI hash) it never rides on AOI rasters: the layer is read live from
# the land-cover set at query time, so a regeneration needs no AOI rebuild.
NLCD_HASH_TAG = 'SNOWTOOL_NLCD_HASH'

# Percent forest cover is stored as a uint8 0..100 with 255 nodata (the same
# integer/nodata convention as the aspect-majority terrain layer).
FOREST_PCT_NODATA = 255

# NLCD land-cover classes counted as "forest" for the percent-forest layer:
# 41 deciduous, 42 evergreen, 43 mixed. Add 90 (woody wetlands) here to count
# forested wetlands as forest.
FOREST_CLASSES = (41, 42, 43)

# Feet <-> meters, for elevation-band math.
M_TO_FT = 3.28084
FT_TO_M = 0.3048

# The elevation range (metres) that elevation bands are generated across, shared
# by every dataset. Bands must be comparable across AOIs *and* across datasets,
# so the bracket is a single global constant rather than a per-dataset DEM range:
# the AOI is the geographic unit of interest, not the dataset. The values only
# need to bracket the highest/lowest terrain any AOI can reach -- they are floored
# to whole `band_step_ft` bins by the elevation BandedZoning, so a generous bracket
# costs nothing but a few empty (null) bands at the extremes. CONUS spans roughly
# Badwater (-86 m) to Mt. Whitney (4421 m); this brackets it with headroom.
# (Resampling onto the ~1km grids only pulls extremes inward, never outward, so a
# bracket valid for a source DEM can never be exceeded by a resampled one.)
MIN_ELEVATION_M = -100.0
MAX_ELEVATION_M = 4500.0
