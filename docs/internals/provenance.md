# Provenance and staleness

Every derived artifact a snowdb writes is expensive to produce and cheap to get
wrong: a burned AOI raster, a terrain or land-cover layer set, a date's ingested
COGs each cost a reprojection, a large source read, or a multi-hundred-megabyte
download. Rebuilding them unconditionally on every command would be wasteful;
trusting whatever is already on disk would be wrong the moment an input changes.
snowtool splits the difference by stamping each artifact with a **provenance
tag** that captures exactly what it was built from, and reducing "is this still
current?" to a single string comparison against what it *would* be built from
now. The machinery is small, and every downstream check is one equality test.

The tag is a **versioned hash**: `v{format_version}:{sha256}`, built by
`versioned_hash(version, digest)` in `snowdb/provenance.py`. The digest covers
the artifact's content or source geometry; the prefix is the *on-disk format
version* of the artifact the hash guards. Storing the two together is the whole
trick. A content change moves the digest, so the tag no longer matches — but a
change to how the artifact is *encoded* on disk (a compression switch, a
boolean-mask-to-cell-area redesign, a new band layout) moves nothing in the
digest, yet is just as much a reason to rebuild. Folding the format version into
the same string means one equality check catches both: bump the producer's
version and every artifact written under the old one reads as stale, forcing a
rebuild, even though its underlying digest is unchanged. All versions start at
`1` — this is a greenfield database, so there is no legacy data to stay
compatible with.

The format version is owned by whatever *produces* the artifact, never
centralized. A zone-layer provider carries its own `format_version` (see
`ZoneLayerProvider` in `snowdb/zones/zone_layer.py`); the AOI-raster writer owns
`AOI_RASTER_FORMAT_VERSION` in `snowdb/aoi_raster.py`, because an AOI raster has
no ingester or provider — the `Dataset` burns it generically; and ingested COGs
carry `INGEST_FORMAT_VERSION` from `snowdb/dataset.py`. Each producer bumps its
own version on a material change to *its* output format, and only artifacts of
that kind go stale.

## The tags

Four hashes and one non-hash tag do the work. The `SNOWTOOL_*` provenance tags
are declared with their rationale in `snowdb/constants.py`; the ingest source
hash is spelled in `snowdb/raster/cog.py` because it is a source-record tag, not
one of the geometry tags.

| Tag | Digest of | Stamped on | Drives |
| --- | --- | --- | --- |
| `SNOWTOOL_AOI_HASH` | basin polygon WKB sha256 | the AOI raster | `pourpoint rasterize` rebuild when missing or stale |
| `SNOWTOOL_DEM_HASH` | generated mean-elevation array sha256 | every layer of a terrain set | terrain-set reconciliation; stale-format findings |
| `SNOWTOOL_NLCD_HASH` | generated percent-forest array sha256 | every layer of a land-cover set | land-cover reconciliation; stale-format findings |
| `SOURCE_HASH` | the source artifact's bytes (sorted, multi-file) | every COG of an ingested date | per-date ingest skip |
| `SNOWTOOL_TILE_BBOX` | *(not a hash — a tile bounding box)* | the AOI raster | AOI window resolution (see below) |

`SNOWTOOL_AOI_HASH` is the versioned hash of a pourpoint's basin geometry.
`Pourpoint.geometry_hash` (`snowdb/pourpoint.py`) is the sha256 of the polygon's
canonical little-endian WKB — only the basin polygon, never the point or the
source properties, because only the polygon affects the burned raster —
and `aoi_provenance` wraps it with `AOI_RASTER_FORMAT_VERSION`. When a
`Dataset` rasterizes an AOI it decides whether to rebuild by reading just that
one tag off the existing COG (`Dataset.aoi_raster_hash`, a header-only
`tags()` read with no array decode) and comparing it to the current
`aoi_provenance`. A changed basin *or* a format bump makes them differ, and
`rasterize_aoi_if_needed` re-burns; a match skips the work entirely.

`SNOWTOOL_DEM_HASH` is the versioned hash of a terrain generation's
mean-elevation array, stamped identically on every layer the pass writes
(elevation, aspect components, aspect-majority). Because it is the same on every
layer, any present layer carries it, so `ZoneLayerSet.provenance_hash` reads it
from just the first. It identifies the DEM a whole terrain set was derived from,
so a set can be reconciled against its source. `SNOWTOOL_NLCD_HASH` is the exact
land-cover analogue over the percent-forest array. The single generation digest
is computed once for the whole streaming pass and then stamped —
`finalize_and_stamp` in `snowdb/zones/generate_common.py` digests each target's
name plus its finalized array in sorted order, turns that into a versioned hash,
and writes it onto every output — so everything produced together reconciles as
one set.

`SOURCE_HASH` guards ingest. `hash_files` (`snowdb/provenance.py`) computes a
single streaming sha256 over the source artifact's bytes, reading each file in
1 MiB chunks (a SNODAS tar can be hundreds of megabytes) and digesting the files
in sorted order so a date built from many tiles is independent of iteration
order. Wrapped with `INGEST_FORMAT_VERSION`, it rides on every COG of the date.
The per-date skip in `Dataset.write_date_cogs` reads it back (again header-only,
via `Dataset._date_source_hash`) and rebuilds unless it matches. This closes a
gap the filenames alone cannot: source filenames embed provenance, so a
*renamed* re-release is already caught by a name mismatch, but a re-release under
the *same* filename with different bytes would keep the names identical — the
hash catches that. `dataset ingest` is therefore converge-by-default: a date
whose COGs carry the current source hash is left untouched; `--force` bypasses
the check and rebuilds regardless.

`SNOWTOOL_TILE_BBOX` is not provenance at all — it rides on AOI rasters to
record the grid-tile bounding box the burned window spans (`ul_row ul_col
br_row br_col`), which the reader uses to resolve the window's origin and tiles.
It is documented with the AOI raster itself; see [AOI rasters and
rasterization](pourpoints.md).

## Why AOI rasters carry area, not elevation

The most consequential provenance decision is what an AOI raster *does not*
contain. A burned AOI raster holds per-pixel geographic cell area (m²) inside
the basin and `0` outside — it is simultaneously the in/out-of-basin membership
mask and the area weights the zonal reduction needs, with no separate area
raster. It carries no elevation, aspect, or forest-cover values. Those are read
live from the terrain and land-cover sets at query time and crossed against the
AOI mask then, not baked in when the basin is burned.

That decoupling is deliberate, and its only AOI-side provenance axis is the
geometry. The cell areas are a pure function of the fixed grid, so they add
nothing to hash. Because elevation and land cover live in their own sets under
their own tags, regenerating a DEM or the NLCD layer never invalidates a single
AOI raster, and changing a basin polygon never forces a terrain or land-cover
regeneration. The two sides move independently: a `SNOWTOOL_DEM_HASH` change
reconciles the terrain set alone; a `SNOWTOOL_AOI_HASH` change re-burns the AOI
raster alone. Query results reflect whichever terrain and land-cover sets are
current at read time, decoupled from when any basin was last rasterized.

## Checking staleness in practice

Every staleness check in the codebase is a header-only tag read — `rasterio`'s
`tags()` without decoding the array — followed by one equality test, so the
guard is orders of magnitude cheaper than the work it may avoid. The AOI check
runs in the rasterize path (`Dataset.aoi_raster_is_current`); the source-hash
check runs in the ingest skip (`Dataset.write_date_cogs`); the format-version
check surfaces in diagnostics. `stale_format_zone_layers` in
`snowdb/diagnostics.py` compares each built zone-layer set's stamped format
version (via `ZoneLayerSet.stored_format_version`) against the provider's
current one and emits a `ZoneLayerFormat` finding for any mismatch, so
`snowtool doctor` flags a set that needs regenerating after a format bump.
`aoi_health_report` separately catches an AOI raster missing its
`SNOWTOOL_TILE_BBOX` tag and points at a rebuild.

`parse_format_version` (`snowdb/provenance.py`) returns the integer version from
a tag, or `None` for a missing value or one not in `v{int}:{digest}` form — an
untagged or legacy artifact. It deliberately does not decide what that means:
the caller does. `ZoneLayerSet.format_is_current` treats a built-but-untagged
set as stale (stored `None` never equals a real version, so it is flagged for
rebuild); a set that is not built at all reports `None` (nothing to check, and
`missing_artifacts` already reports absence). The ingest and AOI reads treat a
missing tag the same way — as stale — so a pre-tagging artifact always rebuilds
rather than being trusted.

The payoff is convergence. Because the expensive operations — a ~1.5 GB NLCD
read, a multi-dataset terrain reprojection, per-date COG builds — are each
gated behind a tag comparison that costs a header read, re-running any command
is safe and nearly free when nothing has changed, and does exactly the work
required when something has. Re-run `generate-zones`, re-run `ingest`, re-run
`rasterize`: each converges to the current inputs without redoing settled work.

For the artifacts these tags guard, see [AOI rasters and
rasterization](pourpoints.md), [zone-layer generation](zone-generation.md), and
[the ingest pipeline](ingest.md); for where the versions and links are
configured, see [Configuration in depth](configuration.md) and [on-disk
layout](on-disk-layout.md).
