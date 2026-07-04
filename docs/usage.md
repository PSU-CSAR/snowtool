# Usage

The CLI is organized into command groups. Run `snowtool --help` or `snowtool
<group> --help` for details, and see the [CLI reference](reference/cli.md) for
the full, generated command listing.

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

Serve the read-only API with uvicorn:

```console
uvicorn snowtool.api.app:get_app --factory
```

`snowtool api serve` wraps the same app for production use. The served API
exposes interactive docs at `/docs` (Swagger UI) and `/redoc`, and its schema
at `/openapi.json`; the same schema is rendered in the [HTTP API
reference](reference/http-api.md).
