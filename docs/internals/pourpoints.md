# Pourpoints, basins, and AOI rasters

Every query snowtool answers is scoped to a basin, and every basin enters the
system as a **pourpoint**: the catalog entity in `snowdb/pourpoint.py`. A
pourpoint is a monitoring or forecast point — a station triplet plus the lon/lat
outflow **point** through which its basin drains — carrying an *optional*
delineated upstream **basin polygon**. The point is always present and is the
pourpoint proper; the basin polygon may be absent (a point-only pourpoint), and
when present it is the thing every downstream artifact is built from. A
pourpoint also keeps the verbatim source properties it was parsed from, a
documented exception to the project's typed-modeling default: this is external,
open-shaped AWDB/USGS data, so only a curated few fields (`awdb_id`, `usgs_id`,
the display `name`) are pulled out as attributes and the rest is carried, never
validated.

Pourpoints are parsed from GeoJSON by `Pourpoint.from_geojson`, which accepts
either a `Feature` with a `Point` geometry (a point-only pourpoint) or a
two-geometry `GeometryCollection` pairing the point with a `Polygon` or
`MultiPolygon` basin. The station triplet is the GeoJSON `id`; it is the
pourpoint's identity. Any unreadable source — garbage bytes, malformed JSON, a
schema mismatch — is classified as a single `GeoJSONValidationError` rather than
a raw decode error, so one bad file in an `import`/`sync` batch lands in that
run's `invalid` list instead of aborting the whole run.

## Triplets, names, and the record store

A station triplet (`snowdb/triplet_naming.py`) is the pourpoint's stable key,
but `:` is not path-safe, so every on-disk artifact keyed by a pourpoint encodes
the triplet with `_` in its filename stem. This one codec —
`triplet_to_stem` / `stem_to_triplet` — is shared by the pourpoint record files
and the per-dataset burned rasters, and both must agree on it for the
`pourpoint sync` prune diff and the raster cascade to line up. The encoding is
lossless because a valid triplet never contains `_`; it is storage naming, not a
type, which is why it lives in its own module rather than in `snowtool.types`.

The lossless source of truth is the **record store**: one GeoJSON file per
pourpoint under `pourpoints/records/<stem>.geojson`, copied verbatim from the
import source (`import`/`sync` never re-serialize a record, they `atomic_copy`
it). A record is written named for its own triplet, so the filename is
authoritative and set diffs can read triplets straight off the filenames without
parsing any geometry.

Three derived quantities hang off the basin polygon and are computed once per
`Pourpoint` (they are `cached_property`, since a pourpoint is treated as
immutable after construction). The `geometry` is the shapely shape of the
polygon; `area_meters` is its geodesic area on the WGS84 ellipsoid, computed
straight from the stored lon/lat polygon via a `pyproj` `Geod` so it is
unit-correct regardless of whatever `basinarea` the source claimed; and
`geometry_hash` is a stable sha256 of the polygon's canonical little-endian WKB.
Only the basin polygon feeds the hash — not the point, not the properties —
because the hash exists to be exactly the signal that a rebuild of the burned
raster is needed, and only a changed basin changes that raster. All three raise
if the pourpoint has no basin.

## Why "pourpoint" record-side and "AOI" raster-side

The codebase deliberately maintains a naming split. Everything record-side —
storage, the index, the CLI `pourpoint` group, the API `/pourpoints` — is
"pourpoint," because that is the catalog entity a user imports and lists. The
term **AOI** survives only in the raster machinery (`AOIRaster`,
`rasterize_aoi`, the `SNOWTOOL_AOI_HASH` tag), and it survives there for a
reason: what gets burned onto a dataset grid is the *basin polygon*, not the
pourpoint. A point-only pourpoint has no AOI to burn. Keeping the two words
apart keeps the distinction legible — "pourpoint" names the record you manage,
"AOI raster" names the per-dataset area-of-interest artifact derived from its
basin.

## The pourpoint index

Parsing every record just to list pourpoints would be wasteful — each basin is
thousands of coordinate pairs — so `snowdb/pourpoint_index.py` maintains a
derived, rebuildable manifest at `pourpoints/index.geojson`. The
**pourpoint index** is a GeoJSON `FeatureCollection` with one `Point` `Feature`
per basin-bearing pourpoint: `id` is the triplet, `geometry` is the outflow
point, and `properties` carries the display name, the geodesic `area_meters`,
the per-dataset coverage map, and the basin `geometry_hash`. Point-only
pourpoints are skipped — with no basin they have nothing to cover and no hash to
index. It is GeoJSON-native on purpose: the exact same file is a plottable point
layer and the FastAPI `/pourpoints` listing payload. Being fully derivable from
`records/`, it never has to be trusted as primary data — a corrupt or
foreign-shaped feature fails loudly on load, and the fix is always a rebuild.
The `geometry_hash` rides in the manifest as an internal rebuild signal and is
not surfaced by the API.

Maintenance splits two ways. `import`, `sync`, and `remove` update the index
**incrementally**: `SnowDbManager._update_index` walks the surviving record
files, indexes a just-parsed pourpoint from memory, reuses an existing entry
as-is while its record and the registered-dataset set are unchanged, and
re-parses from disk only the entries that changed — a self-healing fallback.
`pourpoint reindex` (`PourpointIndex.build`, called with no `reuse`/`preparsed`)
is the explicit **full rebuild** that ignores the persisted index entirely; it
is the recovery path for out-of-band edits to `records/` and for the one change
the incremental path cannot see — a grid change to an already-registered
dataset name, which alters coverage without touching any record.

## Coverage

Whether a dataset can serve a basin at all is a static, per-pourpoint
per-dataset fact, computed in `snowdb/coverage.py`. It varies per dataset
because each dataset has its own grid, CRS, and extent: INSTARR is a
MODIS-sinusoidal western block, while SNODAS and SWANN are geographic national
grids, so a basin fully served by one may be only partially — or not at all —
inside another. The `Coverage` enum has three states: `FULL` (the domain covers
the whole basin — the only state a zonal query may run over without clipping),
`PARTIAL` (the basin overlaps the domain but spills outside it — a query would
silently use only the in-domain portion), and `NONE` (the basin is entirely
outside — an empty mask).

`dataset_coverage` is the pure kernel, and it is reprojection-correct: the basin
(stored WGS84) is moved *into* the domain's CRS before the containment test, so
the test is exact even for a projected grid like MODIS sinusoidal. It uses
shapely `covers` (not `contains`) for `FULL`, so a basin lying exactly on the
domain boundary still counts as fully covered. The domain itself is a
`CoverageDomain` — a polygon in the grid's own CRS. It defaults to the full
grid-extent rectangle but may instead be a dataset's declared **footprint**: the
region it actually serves, e.g. a MODIS block minus a tile that is never
populated, so a basin over a *static* nodata hole is not mis-reported as fully
covered. Per-date data gaps — clouds, a missing day's tile — are deliberately
*not* part of this static geometric domain; they are a separate, per-result
concern the [query engine](queries.md) handles.

## The AOI raster

An **AOI raster** (`snowdb/aoi_raster.py`) is a basin polygon burned onto a
dataset grid, stored at `data/<name>/aoi-rasters/<stem>.tif`. Its one defining
trick is that a single raster is both the membership mask *and* the area
weights: each pixel whose centre falls inside the basin holds that pixel's
geodesic cell area in m² (a `float32`), and every pixel outside holds `0`. Since
no real cell has zero area, `0` doubles as the nodata sentinel — `array > 0` is
the in-basin test and the same values are the weights the zonal reduction needs,
with no separate area raster. The raster carries no elevation, DEM, or terrain
values at all: elevation, aspect, and forest cover are read live from the
[zone layers](zones.md) at query time, so the AOI raster stays decoupled from
them and a terrain rebuild never invalidates it. How the query consumes these
weights is covered in [queries](queries.md).

An AOI raster covers only the tiles its basin spans, not the whole grid. The
window is recorded in the `SNOWTOOL_TILE_BBOX` tag as four space-separated ints,
`ul_row ul_col br_row br_col` — the inclusive tile bounding box. On read
(`tiles_from_tags`) the upper-left tile is the window origin and every tile in
the box is read back; a burned raster missing this tag is treated as a
server-side integrity failure to be fixed by re-rasterizing, not a client error.
The raster is also stamped with `SNOWTOOL_AOI_HASH`, a versioned hash combining
the basin `geometry_hash` with the AOI writer's format version; a raster is
stale when that tag no longer matches the basin's current versioned hash — a
changed basin *or* a format bump — which a cheap tag-only read detects without
decoding the array. See [provenance](provenance.md) for the versioned-hash
mechanism itself.

## How rasterization works

`Dataset.rasterize_aoi` reprojects the basin from WGS84 into the grid's CRS,
computes the clamped tile window with `bounding_tiles` (a basin straddling a
grid edge burns only its in-grid portion; a basin entirely off-grid raises
rather than producing an inverted window), and hands off to `write_aoi_raster`.
The burn is
`rasterio.features.rasterize` with all-touched left off, so a pixel counts as
inside exactly when its centre falls within the basin polygon, producing a
boolean mask. Cell area is then multiplied in: a projected grid uses its single
constant cell area for every pixel, while a geographic grid computes geodesic
area per window row (area depends only on latitude) from the base grid and
broadcasts it across the columns. The mask picks area inside and `0` outside,
and the result is written as a COG tagged with the tile bbox and the AOI
provenance hash.

Rasterization is driven at two granularities. `dataset create` stages a new grid
and rasterizes every indexed basin onto it in one pass (`stage_dataset` →
`rasterize_aois`), gating each burn by coverage so an off-grid (`NONE`) basin is
skipped rather than attempted. The per-pourpoint `pourpoint rasterize` command
rebuilds only what is **missing or stale**: `Dataset.rasterize_aoi` skips a
raster whose `SNOWTOOL_AOI_HASH` already matches and rebuilds otherwise, with
`--rebuild` forcing a byte-level rebuild regardless. Importing or syncing
pourpoints does *not* rasterize — `import`/`sync` only write records and update
the index; rasters are (re)built by `dataset create` staging or an explicit
`pourpoint rasterize`. Removing a pourpoint, by contrast, does cascade:
`_remove_pourpoint_files` deletes the record and every dataset's burned raster
for that triplet. Because a raster is burned once per registered dataset —
including inactive ones — activating a dataset later is instant: its AOI rasters
already exist.
