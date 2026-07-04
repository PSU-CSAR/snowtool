# Walkthrough: standing up a snowdb

This walks through building a snowdb from an empty directory to a running API,
with all three built-in dataset types — **SNODAS**, **UA SWANN (800 m)**, and
**INSTARR/SPIRES** — each carrying its full set of variables.

The order matters: pourpoints are imported and indexed *first*, because each
dataset bakes an AOI raster of every indexed basin onto its grid when it's
created.

!!! note "Network access"
    Creating a dataset generates its zone layers — terrain from USGS **3DEP**
    and land cover from **Annual NLCD** — by reading public cloud data by
    default (NLCD is requester-pays). To work fully offline, supply local
    inputs with `--source PROVIDER PATH` (shown below). Ingesting data also
    requires source archives you provide.

## 1. Initialize an empty snowdb

```console
snowtool snowdb init /srv/snowdb
export SNOWTOOL_SNOWDB_CONFIG=/srv/snowdb
```

`init` lays out the root config (`snowdb_conf.json`), `pourpoints/`, and
`data/`. Exporting `SNOWTOOL_SNOWDB_CONFIG` means later commands don't each
need `--config` (see [Configuration](configuration.md)).

## 2. Import pourpoints

Pourpoint records come from the
[PSU-CSAR/BAGIS-pourpoints](https://github.com/PSU-CSAR/BAGIS-pourpoints)
repository — one GeoJSON file per pourpoint (a station triplet, an outflow
point, and optionally a delineated basin polygon), under `reference/`. Clone it
yourself; don't assume a copy is already present:

```console
git clone https://github.com/PSU-CSAR/BAGIS-pourpoints
snowtool pourpoint import BAGIS-pourpoints/reference
snowtool pourpoint reindex
```

`import` additively imports the basin-bearing pourpoints and skips point-only
ones (a basin polygon is what gets burned into each dataset's AOI raster).
`reindex` rebuilds the `index.geojson` manifest so the datasets you create next
can see the imported basins.

## 3. Create the three datasets

Each `dataset create` stamps a built-in template — carrying all of that
dataset's variables — then stages the dataset's directory, area raster, and
zone layers, and rasterizes every indexed basin onto the new grid. `--activate`
registers it in the root config so readers (and the API) can see it.

```console
snowtool dataset create snodas  --template snodas     --activate
snowtool dataset create swann   --template swann-800m --activate
snowtool dataset create instarr --template instarr    --activate
```

The three template names are `snodas`, `swann-800m`, and `instarr`. The dataset
name (first argument) is yours to choose; it's what you'll ingest and query
against.

To generate zone layers from local files instead of the network defaults, add
`--source` (repeatable) to each `create`:

```console
snowtool dataset create snodas --template snodas \
  --source terrain /data/dem.tif \
  --source landcover /data/nlcd.tif \
  --activate
```

## 4. Ingest source data

Templates define *what* each dataset holds; ingest fills it with dates. Point
`dataset ingest` at one or more source archives for that dataset (see the
dataset's source notes for where to obtain them and their expected format):

```console
snowtool dataset ingest snodas  /data/snodas/*.tar
snowtool dataset ingest swann   /data/swann/*.nc
snowtool dataset ingest instarr /data/instarr/*.h5
```

Ingest is converge-by-default: a date whose COGs already carry the same source
hash is left untouched; a re-release under the same filename with changed bytes
rebuilds. Use `--force` to rebuild every date regardless.

## 5. Validate and start the API

Check the database, then confirm the API app builds before serving:

```console
snowtool snowdb status      # per-dataset artifacts, date span, health
snowtool snowdb validate    # rolls up health checks; non-zero exit on failure
snowtool api serve --check  # validate settings + that the app imports, then exit
```

Then serve it:

```console
snowtool api serve
```

The API opens the catalog once at startup, so if you register or ingest more
after it's running, restart it to pick up the changes. With the server up,
validate interactively:

- `GET /pourpoints` — the imported pourpoints.
- `GET /datasets` — `snodas`, `swann`, `instarr`.
- Browse `GET /docs` (Swagger UI) or `GET /redoc`, and per-dataset stats
  endpoints, all described in the [HTTP API reference](reference/http-api.md).
