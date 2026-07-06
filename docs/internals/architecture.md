# Architecture

The Internals section documents how snowtool is built rather than how it is
driven — the object model behind the CLI and the API, the on-disk shape it reads
and writes, and the machinery (provenance, zoning, ingest, queries) that turns a
directory of COGs into per-basin statistics. This page is the map: the three
entrypoints, the core objects they share, and the read/write/cache split that
keeps them honest. The sibling pages drill into the pieces named here — the
[on-disk layout](on-disk-layout.md), the [configuration surface](configuration.md),
[provenance and staleness](provenance.md), [pourpoints and AOI rasters](pourpoints.md),
the [zone-layer framework](zones.md) and its [generation engines](zone-generation.md),
[ingest](ingest.md), and the [query engine](queries.md).

## Three surfaces, one core

snowtool splits cleanly into three parts that all sit on the same objects. The
**domain core** lives in `snowtool/snowdb/` and knows how to open a database,
read its rasters, and mutate it. The **CLI shell** in `snowtool/cli/` is a thin
wrapper: each command resolves a core object, calls one method, and renders the
result — new logic belongs on the core, never in a click callback. The **read
API** in `snowtool/api/` is a FastAPI app exposing the same reads over HTTP. The
CLI and the API are peers over one core, not layers over each other; neither
imports the other, and the API depends on a pydantic `Settings` the CLI never
touches.

## The read / write / cache split

The core's central object is `SnowDb` (`snowdb/db.py`): the lean, read-only
**catalog** of a database. Built from a root
[`RootConfig`](configuration.md), it binds every registered dataset to its
directory whether or not that directory exists — a dataset is defined by its
config, and a missing directory just means no data yet — so the read path
tolerates an un-initialized root. `SnowDb` holds only constants (config, paths,
specs, datasets, the pourpoint index) and cache-free disk reads. It carries no
mutation methods and no live raster state.

Two siblings wrap a `SnowDb`, each owning one lifecycle the catalog leaves out.
`SnowDbManager` (`snowdb/manager.py`) is the **write surface**: every operation
that mutates the database — creating the layout, registering and activating
datasets, importing and rasterizing pourpoints, generating zone layers, ingesting
data — is a method here, reachable as `manager.db` for the reads it needs. The
inversion is deliberate: *the management layer has a snowdb, not the other way
around*, so the read path can be constructed and served without ever pulling in
the write code. `SnowDbReader` (`snowdb/reader.py`) is the **cached read
surface**: it has a `SnowDb` (as `reader.db`) and owns the one piece of
non-constant read-path state the catalog deliberately lacks — a
[`TiffCache`](#the-raster-read-path) shared across all of a database's COG reads —
plus `zonal_stats`, the [query](queries.md) entry point that is the cache's sole
consumer. Keeping the cache in exactly one type makes test isolation a
type-level fact: a fresh reader is a fresh cache.

The manager and the reader are siblings over the same catalog, not nested in each
other, because they split by lifecycle. The catalog is loop-agnostic and
buildable anywhere. The reader's cache is loop-affine — `alru_cache` binds its
in-flight tasks to the event loop that first awaits them — so a reader must be
built inside the event loop that will use it, while a manager has no such
constraint. That constraint drives where each object is constructed.

## Where each is constructed

The API builds its objects once, at app-lifespan scope. `get_app`
(`api/app.py`) opens a single catalog `SnowDb` from `settings.snowdb_config` and
registers it as a gazebo provider; catalog-only routes (`/`, `/datasets`,
`/pourpoints`) inject that `SnowDb` directly, while the stats routes inject an
app-scoped `SnowDbReader` whose loop-affine cache is born in the app's event loop
at lifespan. Because the catalog is opened once at startup, registering,
activating, or ingesting more while the server runs requires a restart to take
effect. There is no module-level `app`: the ASGI server calls the factory
(`uvicorn snowtool.api.app:get_app --factory`), so importing the module has no
side effects and needs no config.

The CLI builds its objects per invocation. The root `cli` group seeds a
`CliContext` on click's `ctx.obj`; a command that needs the database takes the
`pass_snowdb` decorator (which hands it the lazily-opened `SnowDb`) or
`pass_manager` (which wraps that same `SnowDb` in a `SnowDbManager`). The open is
lazy on purpose — `--version`, `--help`, and the `api` group must never construct
a `SnowDb`, since that would demand a `--config` for commands with no business
touching the database. A write command runs its whole body inside one
`asyncio.run`, so the reader it may build lives in that loop.

## Injection seams

Two categories of pluggable code reach the core by injection rather than as
module globals, so tests can substitute local inputs for network sources without
monkeypatching. **Dataset specs** — the definitions of dataset kinds — arrive as
a `RootConfig` the `SnowDb` is built from; the built-in
`DEFAULT_DATASET_SPECS` back the CLI's `dataset create` templates and the test
fixtures rather than being passed to a `SnowDb` directly. **Zone-layer
providers** (terrain, land cover) are passed to the `SnowDb` constructor,
defaulting to `DEFAULT_ZONE_LAYER_PROVIDERS`; the CLI threads its own set through
`CliContext.zone_layer_providers`. The *sources* those providers read during
generation are injected too — declared in the config, resolved to a
`ZoneLayerSource` per provider, and overridable per command — so a test binds a
`LocalFile` where production reads 3DEP or NLCD over the network. The
[zones](zones.md) and [generation](zone-generation.md) pages own the details;
the point here is that the seams exist so no `SnowDb`, `SnowDbReader`, or
`SnowDbManager` ever reaches for a global to decide where data comes from.

## The raster read path

Raster I/O uses rasterio, async-tiff, and griffine — there is no system GDAL and
no osgeo bindings. A query reduces a `RasterCollection` of dated COGs over an AOI
raster; the actual bytes are read by `TiledRaster.load_tiles`
(`snowdb/raster/tiled.py`), which hands a batch of tile coordinates to async-tiff
together (letting it coalesce the byte-range reads) and then decodes the returned
tiles concurrently with `asyncio.gather`. Across a whole query, tile reads fan
out over many COGs at once.

Opening a COG parses its IFD metadata, so re-opening one for every tile would be
wasteful. `TiffCache` (`snowdb/raster/tiff_cache.py`) is a bounded, async LRU
cache of open async-tiff `TIFF` handles that dedupes concurrent cold opens: the
first request for a key opens the file while the rest await the same in-flight
task, the cache is capped at `maxsize` entries, and a failed open is not cached.
It is an `alru_cache` built fresh in `__init__` — not a module-level decorator —
so each instance owns an independent cache, and exactly one instance is held by a
`SnowDbReader` and shared across that database's reads. Because the underlying
`LocalStore` holds no file descriptors, the cache bounds only retained IFD
metadata, not open fds.

!!! note
    The cache's in-flight tasks bind to the event loop that first awaits a get,
    so a single `TiffCache` — and therefore a single `SnowDbReader` — must be
    used from one event loop only. This is why the reader is constructed inside
    the loop that will drive it (the API at lifespan, the CLI inside
    `asyncio.run`) rather than alongside the loop-agnostic catalog.

## Datasets are data, not subclasses

A dataset kind is not a `SnowDb` subclass; it is a `DatasetSpec` (`snowdb/spec.py`)
carrying a grid, its variables, and an ingester. The built-in specs live in
`snowdb/datasets/` and are collected into `DEFAULT_DATASET_SPECS`. Adding a
dataset is a new spec plus an ingester registered in that package — no new class
in the read path, which stays entirely dataset-agnostic. The [ingest](ingest.md)
page covers how a spec reaches a running database and how source artifacts become
per-date COGs; the [walkthrough](../walkthrough.md) shows the same flow from the
CLI.
