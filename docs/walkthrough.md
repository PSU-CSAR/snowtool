# Walkthrough: standing up a snowdb

This walks through building a snowdb from an empty directory to a running API,
with all three built-in dataset types — **SNODAS**, **UA SWANN (800 m)**, and
**INSTARR/SPIRES** — each carrying its full set of variables.

The order matters: pourpoints are imported and indexed *first*, because each
dataset bakes an AOI raster of every indexed basin onto its grid when it's
staged.

Standing a dataset up is a deliberate register/activate split: **`dataset
create`** stages everything under `data/<name>/` *and* registers the dataset in
the root config — **inactive**, so it exists and is manageable by name but stays
invisible to readers — and **`dataset activate`** later flips it live. Keeping
registration and activation apart lets you stage all three datasets cheaply,
generate their zone layers in a single shared source read (see
[step 4](#4-generate-zone-layers-in-one-shared-pass)), and ingest their data —
so the expensive terrain/land-cover reads happen once for the whole set, and a
dataset is fully populated before anything serves it.

!!! note "Network access"
    Generating a dataset's zone layers reads terrain from USGS **3DEP** and land
    cover from **Annual NLCD** — open public data by default (3DEP tiles over S3,
    and the Annual NLCD bundle over an open MRLC HTTPS download). To work fully
    offline, supply local inputs with `--source PROVIDER PATH` (shown below).
    Ingesting data also requires source archives you provide.

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
point, and optionally a delineated basin polygon), under `reference/`.

`pourpoint sync` mirrors a whole folder of records into the snowdb. Point it
straight at that `reference/` folder on GitHub — the URL your browser shows for
the directory — and every `*.geojson` under it is fetched and imported, no
local clone required:

```console
snowtool pourpoint sync https://github.com/PSU-CSAR/BAGIS-pourpoints/tree/main/reference
```

`sync` imports the basin-bearing pourpoints, skips point-only ones (a basin
polygon is what gets burned into each dataset's AOI raster), prunes any stored
pourpoint absent from the source, and maintains the `index.geojson` manifest as
it goes — the datasets you create next see the imported basins with no separate
reindex step. On this fresh snowdb nothing is pruned; once records exist, a
prune requires `--prune-to <dir>` to archive what it removes.

`sync` also accepts a local directory (after your own `git clone`), and
`pourpoint import` brings in a single record from a local file or a raw file
URL (e.g. a `raw.githubusercontent.com` link).

## 3. Stage and register the three datasets

Each `dataset create` stamps a built-in template — carrying all of that
dataset's variables — stages the dataset's directory and area raster,
rasterizes every indexed basin onto the new grid, and registers the dataset in
the root config **inactive**: it exists and is manageable by name (ingest,
`generate-zones`, diagnostics), but readers don't see it until you activate it
in [step 6](#6-activate-the-datasets). Create never generates zone layers —
the expensive terrain/land-cover reads are inherently deferred to a single
shared `generate-zones` pass in the next step:

```console
snowtool dataset create snodas  --template snodas
snowtool dataset create swann   --template swann-800m
snowtool dataset create instarr --template instarr
```

The three template names are `snodas`, `swann-800m`, and `instarr`. The dataset
name (first argument) is yours to choose; it's what you'll ingest and query
against.

## 4. Generate zone layers in one shared pass

`dataset generate-zones` takes one *or more* datasets and reads each provider's
source **once** over their combined extent, binning it into every one — so
terrain's DEM reproject and the ~1.5 GB NLCD download happen a single time for
the whole set instead of once per dataset. (The NLCD download is kept under
`.cache/landcover/` in the snowdb so later runs skip it; it's safe to delete
once layers are generated.) Activation doesn't matter here —
zone layers live under each dataset's own `data/<name>/`, so the
registered-inactive datasets from step 3 work as is:

```console
snowtool dataset generate-zones snodas swann instarr
```

To generate from local files instead of the network defaults, add `--source`
(repeatable):

```console
snowtool dataset generate-zones snodas swann instarr \
  --source terrain /data/dem.tif \
  --source landcover /data/nlcd.tif
```

## 5. Ingest source data

Templates define *what* each dataset holds; ingest fills it with dates. Each
`dataset ingest` invocation takes a **single source**, whose shape depends on
the dataset (see the dataset's source notes for where to obtain data and its
expected format): a single file for SNODAS (a daily tar archive) and SWANN (a
daily NetCDF) — one file == one date — or a **directory** of SPIRES `.nc`
tiles for INSTARR. Always pass instarr the directory: a date's mosaic is built
from all of its tiles in one ingest call.

```console
snowtool dataset ingest snodas  /data/snodas/SNODAS_20180427.tar
snowtool dataset ingest swann   /data/swann/UA_SWE_Depth_800m_v1_20180427_early.nc
snowtool dataset ingest instarr /data/instarr/
```

Batch driving belongs to the shell. Each ingested date commits via an atomic
whole-directory swap, so parallel runs are safe across distinct dates:

```console
ls /data/snodas/*.tar | xargs -n1 -P4 snowtool dataset ingest snodas
```

This works while the datasets are still inactive — that's the point of the
register/activate split: populate a dataset fully before anything serves it.
Ingest is converge-by-default: a date whose COGs already carry the same source
hash is left untouched; a re-release under the same filename with changed bytes
rebuilds. Use `--force` to rebuild every date regardless.

## 6. Activate the datasets

`dataset activate` flips a registered dataset's `active` flag in the root
config — the visibility toggle that makes it served by the query CLI and the
API (a running API server needs a restart to pick the change up):

```console
snowtool dataset activate snodas
snowtool dataset activate swann
snowtool dataset activate instarr
```

Activation is a pure flag flip — `create` already computed each pourpoint's
coverage when it staged the grid and folded it into the index at registration,
so nothing needs reindexing here.

For a dataset built *out of tree* (its config and data living outside this
snowdb), `dataset add NAME CONFIG_PATH` is the escape hatch that registers an
external config — also inactive, activated the same way. Because `add` skips
staging, pourpoint coverage for that dataset reads as `none` until you run
`snowtool pourpoint reindex`.

## 7. Validate and start the API

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

The API opens the catalog once at startup, so if you register, activate, or
ingest more after it's running, restart it to pick up the changes. With the server up,
validate interactively:

- `GET /pourpoints` — the imported pourpoints.
- `GET /datasets` — `snodas`, `swann`, `instarr`.
- Browse `GET /docs` (Swagger UI) or `GET /redoc`, and per-dataset stats
  endpoints, all described in the [HTTP API reference](reference/http-api.html).
