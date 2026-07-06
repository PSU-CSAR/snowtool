# Datasets and ingest

A dataset kind — SNODAS, SWANN, INSTARR — is **data, not a subclass**. There is
no `SnodasDataset` in the read path; there is a `DatasetSpec` (`snowdb/spec.py`)
that carries one kind's grid, its variables, and an ingester, and a `Dataset`
(`snowdb/dataset.py`) that binds such a spec to a `data/<name>/` directory. The
spec is the path-independent *definition* — it exists with or without any data on
disk — while the `Dataset` is that definition pointed at a place on the
filesystem, owning the per-dataset layout (`cogs/`, `aoi-rasters/`, the
per-provider zone-layer subdirs) and the operations over it. Grid and variables
are always reached through `dataset.spec`.

The spec carries *behavior*, not just settings: given its `GridParams` it builds
the actual griffine grid (`make_grid`), and from that grid's CRS it derives
`is_geographic`, `crs`, and the constant projected `cell_area`. That is
deliberate — a spec is a definition, so it knows how to construct the things its
definition implies rather than being a passive bag handed around for other code
to interpret. The [architecture](architecture.md) page frames the same split
across the whole core; this page covers how a spec reaches a running database and
how a source artifact becomes per-date COGs.

## Specs, the registry, and templates

The built-in specs live one-per-module under `snowdb/datasets/`
(`snodas.py`, `swann.py`, `instarr.py`), each holding its own variables, grid,
and ingester, and are collected into `DEFAULT_DATASET_SPECS`. A `SnowDb` is not
built from those specs directly — it is built from a `RootConfig` (see
[configuration](configuration.md)) — so `DEFAULT_DATASET_SPECS` instead backs two
things: the test fixtures and the `DATASET_TEMPLATES` that `dataset create
--template` stamps into a new dataset's config. A spec round-trips losslessly to
a `DatasetConfig` and back (`config_from_spec` / `DatasetSpec.from_config`), which
is what lets a built-in definition live in exactly one place yet be reproduced as
a config template.

Ingest code is resolved through a small registry. `INGESTERS`
(`snowdb/datasets/__init__.py`) maps an ingester *kind* — `snodas`, `swann`,
`instarr` — to its concrete instance; a dataset config names its ingester by one
of these keys and `DatasetSpec.from_config` looks up the code, raising on an
unknown name. The key is the kind, distinct from a dataset *name*: the built-in
`swann-800m` dataset names the `swann` ingester. Adding a new dataset kind is
therefore a new spec plus an ingester plus one registry entry — no new class in
the read path, which stays entirely dataset-agnostic.

## Dataset variables

A `DatasetVariable` (`snowdb/variables.py`) is one requestable variable and says,
for that variable alone, how to find, read, reduce, and report it. It is a frozen
(hashable) pydantic model and its own persisted form — a config stores variables
as `{key: {unit, reducer, dtype, nodata, glob}}`, with the map key injected back
as `key` on load. Five fields carry the whole contract:

- `glob` finds the variable's single COG inside a `cogs/<YYYYMMDD>/` directory
  (e.g. `*__swe.tif`). COGs are named `<source-provenance>__<key>.tif` and the
  glob anchors on that doubled `__` delimiter, which is why a variable `key` may
  not itself contain `__`.
- `dtype` and `nodata` drive the read: the numpy dtype (`int16`, `uint8`, …) and
  a **finite** fill sentinel. The sentinel must be finite because the stats reader
  masks fill pixels with `values != nodata`, and `x != NaN` is always true — a
  NaN fill would poison every reduction rather than being excluded.
- `reducer` says how the variable aggregates over a basin. `Reducer.MEAN` is the
  area-weighted average over valid pixels; `Reducer.TOTAL` is the area-weighted
  basin total, `Σ(value·area)`, an extensive whole-basin quantity. The
  [queries](queries.md) page owns the reduction math.
- `unit` is a name plus a `scale_factor`, together forming the reported field
  name (`mean_swe_mm`) and the divisor that turns stored units into reported ones.

## The ingest seam

Parsing a source is dataset-*kind*-specific knowledge — a SNODAS tar, a NetCDF, a
tile directory are nothing alike — so, like a spec's variables, it lives on the
spec as an `Ingester` (`snowdb/ingest.py`). `Ingester` is a `Protocol` with a
single `ingest(source, dataset, *, force)` method; the `Dataset` supplies the
generic other half via `write_date_cogs`, which owns the target
`cogs/<YYYYMMDD>/` directory and writes an iterable of `WritableRaster`s into it.
`WritableRaster` is itself a minimal protocol — an `out_name` and a `write_cog` —
so the generic write path never knows any one dataset's input-raster type. The
CLI's `dataset ingest` is thus dataset-agnostic: it resolves the dataset and
calls `dataset.ingest(source)`, which delegates to `spec.ingester` (raising if the
dataset has none). Each run returns an `IngestResult` splitting the dates it
`(re)built` from those it `skipped` as already current; the CLI reports the two
sets on separate lines.

Completeness is enforced at *date* granularity, twice. Before any filesystem work
the rasters an ingester produced must cover every spec variable, so a source short
a required input fails fast with `IncompleteDatasetDataError`; after writing into
the staging directory, every spec variable must again resolve to exactly one COG
or the swap is abandoned and the existing date left untouched.

## Converge by default

Ingest is idempotent per date. Each date is stamped with a **source hash** — a
`versioned_hash(INGEST_FORMAT_VERSION, …)` over a streaming sha256 of the source
bytes (multi-file sources are digested in sorted order; the mechanism lives in
[provenance](provenance.md)) — written into every COG of the date as the
`SOURCE_HASH` tag. `write_date_cogs` skips a date only when its directory already
holds *exactly* the COGs this call would write **and** their stored source hash
equals the new one. The filename set alone is insufficient: source filenames embed
provenance, so a *renamed* re-release is caught by a name mismatch, but a
re-release under the *same* filename with different bytes keeps the names
identical — the hash equality catches that and forces a rebuild. A missing tag (a
date written before hashing) also reads as stale. `--force` rebuilds every date
regardless.

The format version rides inside that same hash, so it doubles as a global
staleness lever.

!!! note
    Bump `INGEST_FORMAT_VERSION` (`snowdb/dataset.py`) only on a material change
    to the *encoding* of an ingested COG — compression, band layout, nodata
    handling — not for source-data changes. A bump makes every existing date read
    as stale (its stored `v{n}:…` no longer matches) and rebuild on the next
    ingest, even though the underlying source bytes are unchanged.

## Atomicity and parallelism

The whole per-date directory is the unit of commit. `write_date_cogs` stages the
date's COGs into a fresh temp directory beside the target and swaps it in
wholesale via `staged_dir` (`snowdb/atomic.py`), so a crash mid-ingest never
leaves a *partial* date — a reader sees the wholly-old directory, a brief gap, or
the wholly-new one — and stale COGs from a prior, differently-named source vanish
by construction instead of lingering beside the new ones and making a variable's
glob ambiguous. Because each date commits independently and atomically, parallel
ingest runs across *distinct* dates are safe, which is what lets the walkthrough
drive a batch with `xargs -n1 -P4`. Batch driving is left to the shell; a single
`ingest` invocation always takes one source artifact.

## The three built-in ingesters

**SNODAS** ingests one daily tar archive — one archive is one date. The archive
holds gzipped raw rasters plus their `.Hdr`/`.txt` headers;
`SNODASInputRasterSet.from_archive` extracts the tar, gunzips each member, and
parses the SNODAS filename of every header into a `SNODASInputRaster`. The
filename regex yields the product code and a `vcode`, which map to the eight
`Product` variables (`swe`, `depth`, `average_temp`, `runoff`, `sublimation`,
`sublimation_blowing`, and precip split by `vcode` into `precip_liquid` /
`precip_solid`); the set refuses an archive missing any product and refuses one
whose dates disagree. Two format quirks are handled: GDAL's SNODAS/raw driver has
a header line-length limit, so each header is trimmed to 255 chars before read,
and ingest pins to the `05` time-step hour — the standard daily product — so a
date never mixes revisions (the parser stays general; a policy gate does the
refusing). Each COG keeps the full parsed source stem as its name, with the parsed
fields stamped as `SOURCE_*` tags.

**SWANN** ingests one UA 800 m daily NetCDF — one file is one date, named
`UA_SWE_Depth_800m_v1_<YYYYMMDD>_early.nc`. The date comes from the filename; the
`SWE` and `DEPTH` int16 subdatasets are opened as `netcdf:<file>:<SWE|DEPTH>` URIs
and written straight out as grid-aligned COGs, one per variable, named
`<source-stem>__<key>.tif`. There is no reprojection or flip: GDAL's NetCDF driver
already returns each array north-up and aligned to the dataset grid, so each band
is written on the spec's authoritative transform/CRS rather than GDAL's
lat/lon-derived float32 geotransform. Like SNODAS, ingest pins to a single
revision — the `_early` processing stage (latency over finality) — refusing
`_provisional`/`_stable` files with a precise error.

**INSTARR** ingests a *directory* of SPIRES NRT tiles, because the unit of the
product is one NetCDF per MODIS tile per day
(`SPIRES_NRT_h##v##_MOD09GA###_<YYYYMMDD>_V1.0.nc`) and a date's mosaic needs all
of its tiles at once — a per-tile call would rebuild the date from a single tile,
last write wins. The ingester scans the directory recursively, groups tiles by
date, and for each date writes one mosaicked COG per variable (nine SPIRES
variables, uint8 or uint16). The mosaic is a **lossless stitch on the native MODIS
Sinusoidal grid, not a reprojection**: `InstarrMosaicRaster` allocates the full
grid filled with nodata, then drops each tile band into its slot positioned by the
tile's own sinusoidal origin (adjacent MODIS tiles abut exactly, so values stay
bit-exact at native 463 m resolution), leaving any absent tile as nodata. Because
the mosaic spans all the date's tiles, the per-tile `h##v##` is dropped from the
COG's distilled provenance stem; a date whose tiles disagree on collection or
version is refused, and the exact contributing tiles are recorded in the COG's
`SOURCE_FILES` tag.

## What an ingested date looks like on disk

An ingested date is a directory `cogs/<YYYYMMDD>/` holding one COG per spec
variable, each named `<source-provenance>__<key>.tif`. Every COG is written by the
shared `write_cog` helper (`snowdb/raster/cog.py`) with the project's fixed
creation options: the GDAL `COG` driver, DEFLATE compression at level 9, an
integer horizontal-differencing predictor (`predictor=2` for ingest), a block
size equal to the dataset's `tile_size`, and *no* overviews (the read path only
ever reads full resolution, so skipping them keeps output smaller and
deterministic). Each COG also carries embedded per-band statistics
(`STATISTICS_MINIMUM`/`MAXIMUM`/`MEAN`/`STDDEV`, computed over non-nodata pixels)
and the `SOURCE_*` provenance tags — dataset, date, variable, contributing files,
kind-specific extras, and the `SOURCE_HASH` the skip check reads back. The
[on-disk layout](on-disk-layout.md) page places `cogs/` within the wider dataset
directory.
