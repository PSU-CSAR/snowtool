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

The CLI is organized into command groups; run `snowtool --help` or
`snowtool <group> --help` for details:

| Group | Purpose |
| --- | --- |
| `snowdb` | Create and manage the snow database. |
| `dataset` | Register and ingest gridded snow datasets. |
| `pourpoint` | Manage pourpoints and their delineated basins. |
| `query` | Zonal statistics and date listings over a snowdb. |
| `report` | Read-only diagnostics and health reports. |
| `api` | Run the read-only HTTP API server. |
| `windows` | Windows-only admin (IIS deployment, all-users PATH). |

### HTTP API

Serve the read-only API with uvicorn:

```commandline
uvicorn snowtool.api.app:get_app --factory
```

`snowtool api serve` wraps the same app for production use.

## Deployment

### Windows / IIS

Install the tool with `uv tool install snowtool`, then deploy it as an IIS
site fronting `snowtool api serve` via httpPlatformHandler:

```commandline
snowtool windows iis install C:\inetpub\snowtool --hostname snow.example.org --config C:\snowdb\snowdb_conf.json
```

Re-running `snowtool windows iis install` against an existing site updates it
in place. Requires, on the target Windows Server: IIS with the
httpPlatformHandler and IISAdministration modules installed, and an elevated
(Administrator) PowerShell/shell. Tear a site down with `snowtool windows iis
remove`.

#### Making `snowtool` available to all users

`uv tool install` and `uv tool update-shell` only ever touch the *installing
user's* profile and PATH — there's no install-time hook to make the tool
available machine-wide. To get `snowtool` onto every user's PATH on a Windows
Server:

1. Point `uv` at a shared install location instead of its per-user default,
   in an elevated shell, before installing:

   ```commandline
   setx /M UV_TOOL_DIR C:\ProgramData\uv\tools
   setx /M UV_TOOL_BIN_DIR C:\ProgramData\uv\bin
   ```

   `setx /M` sets these machine-wide; open a *new* elevated shell so they
   take effect, then run `uv tool install snowtool`.

2. Add the shared bin directory to the machine-wide PATH:

   ```commandline
   snowtool windows add-to-path
   ```

   This must run in an elevated shell. It refuses to proceed (and prints
   these same steps) if it detects a per-user install — e.g. if step 1 was
   skipped and `snowtool` landed under the installing admin's own profile —
   since putting a per-user path on the machine-wide PATH would only work for
   that one account. Open a new shell afterward to pick up the change.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and
project conventions.

Team members: see the [development notes](https://docs.google.com/document/d/1RZVDGtgij7DplTrgpXjPJp1hsV3guXFNomO27nuLHiU/edit?usp=sharing)
for additional context.
