# A snowdb on disk

A snowdb is a plain directory. Nothing about it is a database in the
server sense: it is a tree of JSON configs, GeoJSON records, and
cloud-optimized GeoTIFFs that any process can read directly, and that
`snowtool` reaches entirely by following one **root config** at the top.
This page is the map of that tree — what each file is, who writes it, who
reads it, and which files are the source of truth versus derived artifacts
that can always be rebuilt. The configs themselves, field by field, are
[Configuration in depth](configuration.md); provenance tagging and
staleness are [provenance](provenance.md).

## The tree

A snowdb laid out by `snowtool init` and populated with one
path-linked dataset looks like this:

```text
/srv/snowdb/
├── snowdb_conf.json                root config (snowtool.snowdb/v1) — the entry point
├── pourpoints/
│   ├── index.geojson               derived manifest, rebuildable (`pourpoint reindex`)
│   └── records/
│       └── <triplet>.geojson       source-of-truth pourpoint records
├── data/
│   └── <dataset>/
│       ├── dataset.json            dataset config (snowtool.dataset/v1)
│       ├── cogs/
│       │   └── <YYYYMMDD>/         one directory per ingested date
│       │       └── *__<var>.tif    per-variable COGs for that date
│       ├── aoi-rasters/
│       │   └── <triplet>.tif       per-basin AOI rasters burned onto this grid
│       ├── terrain/                terrain zone-layer set (elevation, aspect, …)
│       └── landcover/              land-cover zone-layer set (forest_cover)
└── .cache/
    └── landcover/                  cached NLCD source download (safe to delete)
```

The only fixed name in the whole system is `snowdb_conf.json`
(`CONFIG_FILENAME` in `snowdb/config.py`): it is what `snowtool init` writes
and what `SnowDb.open` looks for when handed a directory. Everything else
is reached by *following the config* rather than by assuming a path, so
almost every location here is a default the root config could override.

## The root config

`snowdb_conf.json` is the system's single entry point. Handed one path,
`SnowDb` (`snowdb/db.py`) resolves everything else — the datasets, the
pourpoint index and records — from it. It is a small `RootConfig`
(`snowdb/config.py`) written and read only through `snowtool`: `snowtool
init` creates it, and the `dataset` command group edits its `datasets`
map. It is pure source of truth — the record of *which datasets exist* and
*which readers serve* — and carries no derived data, so it is never
regenerated, only edited. Its fields are covered in
[Configuration in depth](configuration.md).

## Pourpoints: records versus index

`pourpoints/` is the catalog, and it is the canonical example of the
source-of-truth/derived split that runs through the whole store.

`records/<triplet>.geojson` are the **pourpoint records**: one lossless
GeoJSON file per pourpoint — a station triplet, an outflow point, and
optionally a delineated upstream basin polygon. These are the source of
truth. `pourpoint import`/`sync` write them (importing a source file
verbatim where possible, via `atomic_copy`), and `pourpoint remove`
deletes them. Their semantics — triplets, the optional basin, how a basin
becomes an AOI raster — are [pourpoints](pourpoints.md).

`index.geojson` is the **pourpoint index**: a derived, rebuildable
manifest (`PourpointIndex` in `snowdb/pourpoint_index.py`). Parsing every
record — each basin is thousands of coordinate pairs — just to list
pourpoints is wasteful, so the index denormalizes the list-relevant fields
into one `Point` feature per pourpoint (`id` = triplet, `geometry` = the
outflow point, `properties` = name, geodesic `area_meters`, per-dataset
coverage, and the basin `geometry_hash` as an internal rebuild signal). It
is deliberately GeoJSON-native: the same file is a plottable point layer
and the API's listing payload. Because it is derived it never has to be
trusted as primary data — if it drifts, `pourpoint reindex` rebuilds it
from `records/` in full. Import, sync, and remove keep it up to date
incrementally; `reindex` is the recovery path for out-of-band edits and
for a grid change to an already-registered dataset.

## Per-dataset data

Each registered dataset owns a directory (`data/<name>/` by convention),
managed by `Dataset` in `snowdb/dataset.py`. For a path-linked dataset the
directory holds a `dataset.json` (`DatasetConfig`, `snowtool.dataset/v1`)
that fully describes the dataset — its grid, variables, ingester, and
zones; for an inline dataset that config lives in the root config instead
and there is no `dataset.json` on disk. Everything else under the
directory is a derived, rebuildable artifact:

`cogs/<YYYYMMDD>/` holds one directory per ingested date, each containing
the per-variable COGs for that date. Files are named
`<source-provenance>__<key>.tif`, and the read path is dataset-agnostic:
it finds a variable's file by the variable's `glob` and takes the date
from the directory name, never parsing filenames. Ingest writes these
(see [ingest](ingest.md)); queries read them.

`aoi-rasters/<triplet>.tif` holds one AOI raster per basin, burned onto
*this dataset's* grid. The raster is both the basin membership mask and
the per-pixel cell-area weights for a query; `pourpoint rasterize` builds
it and the query engine reads it. It is fully derived from a basin polygon
plus the grid, and it is tagged with the basin's geometry hash so a
changed basin reads as stale and rebuilds ([provenance](provenance.md)).

`terrain/` and `landcover/` hold the **zone-layer sets** — the derived
elevation/aspect/forest-cover grids a query crosses its statistics over.
Each configured zone-layer provider writes into its own subdirectory
(`terrain`, `landcover`), named by the provider's `subdir`
(`snowdb/zones/`). A provider's subdirectory exists only for a dataset
whose config `zones` enables it. These are generated by `dataset
generate-zones` and are covered in [zones](zones.md) and
[zone generation](zone-generation.md).

## The source cache

`.cache/` is not database content at all: it holds inputs a
download-and-cache generation source keeps around between runs. Its one
occupant today is `.cache/landcover/`, where the default land-cover source
fetches the ~1.5 GB Annual NLCD bundle and extracts the national GeoTIFF
the forest-cover layer is binned from
([zone generation](zone-generation.md)). The cache sits under the snowdb
root rather than a temp directory deliberately: temp directories are
reaped (on reboot, or by age-based cleaners), which would turn a later
regeneration back into a re-download; the snowdb's volume is the one sized
for bulk raster data, where tmp may be a small or RAM-backed partition;
and scoping it to the root means the cache lives and dies with the
database — deleting the snowdb orphans nothing elsewhere.

It is pure cache: once zone layers are generated it is safe to delete, and
the only cost of deleting it is re-downloading on the next generation run.
A snowdb whose land-cover source is pinned to a local file (the root
config's `sources` map, or `--source landcover PATH`) never creates it.

## Atomic writes

Every persisted file is read back later by another process — a query, a
reindex, the API at startup — so a write that dies partway through (a
crash, `ENOSPC`, `^C`) must never leave a *torn* file where a reader will
find it. The helpers in `snowdb/atomic.py` give every writer that
guarantee with one primitive: write into a uniquely named temp path
*beside* the destination (same directory, so guaranteed same filesystem),
then `os.replace` it onto the destination. A same-filesystem rename is
atomic on POSIX and on Windows, so the swap can never be observed
half-done, and any failure removes the temp file and leaves the
destination exactly as it was. `atomic_write_text` covers the JSON configs
and the index; `atomic_copy` imports a source record byte-for-byte without
re-encoding it.

A whole per-date COG directory is committed the same way through
`staged_dir`: the caller populates a fresh temp directory beside the
target, and on clean exit the old directory (if any) is renamed aside, the
temp directory is renamed onto the target, and the old one is removed.
POSIX has no primitive that swaps two non-empty directories in one step,
so there is a brief, deliberate window between those two renames when
nothing exists at the target — but a reader can never see a *partial*
directory, only the wholly-old tree, a sub-millisecond gap, or the
wholly-new tree. This also means a re-ingest cannot leave stale COGs from a
differently-named source lingering beside the new ones.

!!! note
    This is crash-consistency of *content*, not durability against power
    loss. There is no `fsync`, so a kernel or power failure immediately
    after a successful rename can still lose the write to page cache. What
    ends up on disk is always a complete prior version or a complete new
    version, never a partial one — a deliberate trade-off, since a snowdb
    is a rebuildable store and not a durability-critical one.

## Resource-typed entities

Every persisted config carries an opaque, versioned `resource`
discriminator as its first field: `snowtool.snowdb/v1` for the root config,
`snowtool.dataset/v1` for a dataset config (`ResourceModel` in
`snowdb/config.py`). The `/vN` is human-facing, but the whole string is an
exact-match type tag: a schema change is a *new* type with its own model
and migration, never an in-place reinterpretation of the old one. No
entity's version constrains another's, and there is deliberately no global
snowdb version number. Each entity is loaded by its own type-specific
classmethod (`DatasetConfig.load`, `RootConfig.load`), which parses the file
and wraps any parse failure as a `SnowDbConfigError`; the `resource` field is
still what identifies the file's kind for a reader inspecting it directly.
This is a greenfield store — every resource starts at `v1` and there is no
migration machinery; see [provenance](provenance.md) for how the same
versioned-hash idea tags derived artifacts.

## Relocatability

Because the root config is the one entry point and everything is reached
from it, a snowdb is relocatable as long as its internal links stay
relative. A relative link — a dataset `path`, `pourpoint_index`,
`pourpoint_records`, or a generation `source` — resolves against the root
config's own directory, so moving the whole tree keeps every link valid.
An absolute link points at a fixed location and decouples that piece from
the tree (a dataset staged elsewhere, a shared DEM). Data-directory
conventions follow the same rule: a path-linked dataset's data lives
beside its config wherever the path points, and an inline dataset's data
lives at `data/<name>/` under the root. A `RootConfig` built in code with
no path on disk has no root to resolve against, so it can use only absolute
links and inline datasets — which is exactly how the test suite builds a
whole snowdb with no files at all. The resolution rules are spelled out in
[Configuration in depth](configuration.md).
