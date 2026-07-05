# Usage

The CLI is organized into command groups. Run `snowtool --help` or
`snowtool <group> --help` for details, and see the
[CLI reference](reference/cli.md) for the full, generated command listing.

| Group | Purpose |
| --- | --- |
| `snowdb` | Create and manage the snow database. |
| `dataset` | Register and ingest gridded snow datasets. |
| `pourpoint` | Manage pourpoints and their delineated basins. |
| `query` | Zonal statistics and date listings over a snowdb. |
| `report` | Read-only diagnostics and health reports. |
| `api` | Run the read-only HTTP API server. |
| `windows` | Windows-only admin (IIS deployment, all-users PATH). |

Set [`SNOWTOOL_SNOWDB_CONFIG`](configuration.md) (or pass `--config`) so the
database-backed groups know which snowdb to open.

## HTTP API

Serve the read-only API with `snowtool api serve`:

```console
snowtool api serve --config /srv/snowdb
```

`serve` runs the app under uvicorn, forwarding uvicorn's own options
(`--host`/`--port`/`--workers`/`--reload`/…); add `--check` to validate settings
and that the app imports without starting a server. The served API exposes
interactive docs at `/docs` (Swagger UI) and `/redoc`, and its schema at
`/openapi.json`; the same schema is rendered in the [HTTP API
reference](reference/http-api.html).
