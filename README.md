# `snowtool`

Snow analysis tool for basin forecasting.

`snowtool` manages a **snowdb** — an on-disk directory holding multiple
gridded snow datasets (SNODAS, UA SWANN, SPIRES/INSTARR, …) — and answers
per-basin queries against it: zonal statistics of snow variables crossed over
elevation bands, aspect classes, and forest-cover thresholds. It ships as a
CLI and a read-only HTTP API, both sitting on the same core.

A snowdb is a plain directory: a root config, a `pourpoints/` catalog, and a
`data/<dataset>/` tree of per-date COGs. Basins are delineated from pourpoints
and burned into per-dataset area-weighted masks; queries reduce datasets over
those masks live. Raster I/O is rasterio + async-tiff + griffine — **no system
GDAL required.**

## Installation

Requires Python 3.14+. Install as a [uv](https://docs.astral.sh/uv) tool:

```commandline
uv tool install snowtool
```

Or run from a clone for development — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Configuration

`snowtool` reads its snowdb location from the `SNOWTOOL_SNOWDB_CONFIG`
environment variable (or the per-command `--config`/`-C` option). Dotenv
loading is disabled, so the variable must be set in the environment:

```commandline
export SNOWTOOL_SNOWDB_CONFIG=/path/to/snowdb/snowdb_conf.json
```

## Usage

The CLI is organized into command groups (`snowdb`, `dataset`, `pourpoint`,
`query`, `report`, `api`, `windows`); run `snowtool --help` or `snowtool
<group> --help` for details. Serve the read-only HTTP API with
`uvicorn snowtool.api.app:get_app --factory` (or `snowtool api serve`).

See the [documentation](https://psu-csar.github.io/snowtool/) for full
usage, deployment (including Windows/IIS), and generated CLI, HTTP API, and
Python API reference.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and
project conventions.

Team members: see the [development notes](https://docs.google.com/document/d/1RZVDGtgij7DplTrgpXjPJp1hsV3guXFNomO27nuLHiU/edit?usp=sharing)
for additional context.
