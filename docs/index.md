# snowtool

Snow analysis tool for basin forecasting.

`snowtool` manages a **snowdb** — an on-disk directory holding multiple gridded
snow datasets (SNODAS, UA SWANN, SPIRES/INSTARR, …) — and answers per-basin
queries against it: zonal statistics of snow variables crossed over elevation
bands, aspect classes, and forest-cover thresholds. It ships as a CLI and a
read-only HTTP API, both sitting on the same core.

A snowdb is a plain directory: a root config, a `pourpoints/` catalog, and a
`data/<dataset>/` tree of per-date COGs. Basins are delineated from pourpoints
and burned into per-dataset area-weighted masks; queries reduce datasets over
those masks live. Raster I/O is rasterio + async-tiff + griffine — **no system
GDAL required.**

## Where to go next

- **[Installation](installation.md)** — install as a `uv` tool or from source.
- **[Configuration](configuration.md)** — pointing snowtool at a snowdb.
- **[Usage](usage.md)** — the CLI command groups and the HTTP API.
- **[Internals](internals/architecture.md)** — how it works underneath: the
  object model, the on-disk layout and full config surface, provenance and
  staleness, zone layers and their generation, ingest, and the query engine.
- **[Deployment](deployment/windows-iis.md)** — running the API under IIS on
  Windows.
- **Reference** — generated [CLI](reference/cli.md), [HTTP
  API](reference/http-api.html), and [Python API](reference/python-api.md) docs.
