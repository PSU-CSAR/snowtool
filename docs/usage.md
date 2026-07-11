# Usage

The CLI has a handful of top-level commands plus a few command groups. Run
`snowtool --help` or `snowtool <command|group> --help` for details, and see
the [CLI reference](reference/cli.md) for the full, generated command
listing.

| Command / group | Purpose |
| --- | --- |
| `init` | Create an empty snowdb. |
| `status` | Overview of every registered dataset: active flag, artifacts, date span. |
| `doctor` | Run health checks (grid, dates, files, pourpoints) and exit 1 on any finding. |
| `stats` | Crossed zonal statistics for one pourpoint/dataset, with an OGC `--dates`/`--years` interval. |
| `dataset` | Register, ingest, and inspect gridded snow datasets. |
| `pourpoint` | Manage pourpoints and their delineated basins. |
| `api` | Run the read-only HTTP API server. |
| `windows` | Windows-only admin (IIS deployment, all-users PATH); hidden unless running on Windows. |

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
