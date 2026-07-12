# Configuration in depth

A snowdb has one entry point: the **root config**. Hand `snowtool` a single
path and it reaches everything else — the datasets, the pourpoint catalog,
the generation sources — by following that config, never by assuming where
things live. This page is the full knob surface: every field of the root
config and of a dataset config, how the two fit together with the data on
disk, and how a config reaches the running CLI and API. For the
environment-variable basics of *pointing* snowtool at a snowdb, see
[Configuration](../configuration.md); for how these files sit in the
directory tree they describe, see [A snowdb on disk](on-disk-layout.md).

Every persisted config is a pydantic model in `snowdb/config.py` and
carries an opaque, versioned `resource` tag as its first field
(`snowtool.snowdb/v1`, `snowtool.dataset/v1`). A schema change is a new
`resource` type, not an in-place reinterpretation; there is no global
snowdb version. That tagging, and the atomic writes these files are saved
with, are covered in [A snowdb on disk](on-disk-layout.md).

## The root config

`RootConfig` (`resource: "snowtool.snowdb/v1"`) is written to
`snowdb_conf.json` and is the source of truth for what datasets exist and
where the pourpoint catalog lives.

`created_at` is the UTC timestamp stamped when `snowtool init` created the
root; it is informational.

`datasets` maps a dataset *name* to its **link** — the registration of one
dataset. The map is empty on a fresh snowdb; `dataset create`/`register`
register into it and `dataset activate`/`deactivate` toggle entries. This
map, plus each link's `active` flag, is the whole answer to "what datasets
does this snowdb have, and which does it serve." Links are detailed below.

`pourpoint_index` and `pourpoint_records` are the locations of the
pourpoint manifest and the record directory, defaulting to
`pourpoints/index.geojson` and `pourpoints/records`. They are plain paths,
resolved relative to the root config's directory (or absolute); overriding
them is rarely needed but supported for an unusual layout.

`sources` maps a zone-layer provider name (`terrain`, `landcover`) to a
path that overrides where that provider reads its input during `dataset
generate-zones`. A provider absent from this map uses its network default
(USGS 3DEP for terrain, the MRLC Annual NLCD bundle for land cover). Paths
are absolute or relative to the root config. This is the place to pin an
offline or pre-staged input for the whole snowdb rather than passing
`--source` on every generation run. Generation itself is
[zone generation](zone-generation.md).

## Dataset links: path versus inline, active versus not

A link is a discriminated union on `type`. A **path link** (`type:
"path"`) *references* a dataset config file elsewhere on the filesystem:

```json
{ "type": "path", "path": "data/snodas/dataset.json", "active": true }
```

The `path` is relative to the root config's directory (a relocatable tree)
or absolute (a dataset staged elsewhere). The dataset's data lives beside
its config wherever the path points. This is what `dataset create` writes:
a `dataset.json` under `data/<name>/` and a relative link to it.

An **inline link** (`type: "inline"`) *embeds* the whole dataset config in
the root config instead of pointing at a file:

```json
{ "type": "inline", "active": true, "dataset": { "resource": "snowtool.dataset/v1", "...": "..." } }
```

An inline dataset has no `dataset.json` on disk; its data lives at the
conventional `data/<name>/` under the root. Inline definitions let a whole
snowdb be built in memory with no dataset files at all, which is how the
test suite constructs one. Prefer a path link for a normal on-disk dataset
(its config lives with its data and stays independently editable); reach
for inline for a small, self-contained, or programmatically built snowdb.

The `active` flag gates **visibility to readers**, not existence. An
inactive dataset is still registered — resolved by name for management
(ingest, zone generation, health checks) — but the query CLI and the API
skip it. `dataset create`/`register` register a dataset **inactive** so it can
be fully staged and populated before anything serves it, and `dataset
activate` flips the flag. A bare hand-written link omits `active` and
defaults to `True`, so a config authored by hand just works. The
register/activate split, and why it exists, is walked through in the
[walkthrough](../walkthrough.md); the read-versus-manage split it drives is
[architecture](architecture.md).

## The dataset config

`DatasetConfig` (`resource: "snowtool.dataset/v1"`) is everything a dataset
*is*, independent of where it lives — the same model whether it sits in a
`dataset.json` or inline in the root config. It carries no name: the name
comes from the key it is registered under.

`grid` is a `GridParams` (`snowdb/grid.py`), the definition of the
dataset's north-up tiled grid. It carries the pixel-space origin
(`origin_x`, `origin_y`), the pixel size in CRS units (`px_size`), the grid
size in pixels (`cols`, `rows`), the `tile_size` in pixels per tile edge,
and the `crs` as an EPSG integer or WKT string (default `4326`). A
geographic CRS means per-cell area varies by latitude (the AOI raster burns
geodesic area); a projected CRS means constant cell area. The grid is the
one type used both as the live definition and its persisted form.

`variables` maps a variable *key* to a `DatasetVariable`
(`snowdb/variables.py`). The map key *is* the variable's key and is
injected on load, so it is not repeated inside the value. Each variable
declares `glob` (the filename glob that finds its COG within a
`cogs/<YYYYMMDD>/` directory), `dtype` (the numpy read dtype, e.g.
`int16`), `nodata` (a finite fill sentinel — NaN is rejected, because the
stats reader masks fill with a `!=` compare that can never exclude NaN),
`reducer` (`mean` for an area-weighted intensive average or `total` for an
area-weighted basin total), and `unit` (an inline `{name, scale_factor}`
that reports the value). A key may not contain `__`, which is the COG
filename delimiter between provenance and variable.

`ingester` is the registry name of the code that turns a source artifact
into this dataset's per-date COGs — one of the keys in `INGESTERS`
(`snowtool.snowdb.datasets`): `snodas`, `swann`, `instarr`. `None` means a
read-only/derived dataset with no ingest. The name is the dataset *kind*
(`swann`), distinct from a dataset *name* (`swann-800m`). Ingest is
[ingest](ingest.md).

`zones` maps a zone-layer provider to its layers to each layer's default
query params: `{provider: {layer: ZoneLayerParams | None}}`. A provider's
*presence* enables it for the dataset (its layers are generated and
served); its absence means the dataset has no such zone layer.
`ZoneLayerParams` is a union of four single-field, `extra='forbid'` models
— `BandStepParams` (`band_step_ft`), `BucketParams` (`buckets`),
`ThresholdParams` (`threshold_pct`), `EntropyThresholdParams`
(`entropy_threshold`) — so a block parses to exactly one member by its
field name, and a mistyped or unknown param fails at config load rather
than being silently ignored. A layer enabled with no params (a
categorical axis, e.g. `terrain.aspect`) is stored as `null`, not `{}`; a
param attached to a layer whose scheme doesn't take it (e.g. `buckets` for
elevation) parses fine here but raises `ZoneParamsError` at query time,
when `ZoneScheme.configured` folds it into the layer's actual scheme. The
zoning framework and what each knob does is [zones](zones.md).

`footprint` is an optional GeoJSON geometry, in the grid's CRS, giving the
region the dataset actually *serves* — for example a MODIS block minus a
never-ingested tile. Omitted, the dataset serves its whole grid extent. It
is used for AOI coverage classification, so a basin over a permanently
empty hole is not reported as fully covered. Set it only when the served
region is genuinely smaller than the grid rectangle.

`data_dir` overrides where the dataset's data lives — the directory holding
`cogs/`, `aoi-rasters/`, and the zone subdirectories. Absolute points
anywhere (decoupled from the config's location); relative resolves against
the config's own directory; omitted uses the convention (beside a
referenced config, or `data/<name>/` for an inline one). Leave it unset
unless a dataset's data must sit apart from its config.

## Worked examples

A root config with one path-linked dataset and one inline dataset, pinning
terrain generation to a local DEM:

```json
{
  "resource": "snowtool.snowdb/v1",
  "created_at": "2026-07-05T17:32:00Z",
  "datasets": {
    "snodas": {
      "type": "path",
      "path": "data/snodas/dataset.json",
      "active": true
    },
    "swann": {
      "type": "inline",
      "active": true,
      "dataset": {
        "resource": "snowtool.dataset/v1",
        "grid": {
          "origin_x": -125.0208,
          "origin_y": 49.9375,
          "px_size": 0.008333325394357,
          "cols": 7025,
          "rows": 3105,
          "tile_size": 256,
          "crs": 4269
        },
        "variables": {
          "swe": {
            "unit": {"name": "mm", "scale_factor": 1.0},
            "reducer": "mean",
            "dtype": "int16",
            "nodata": -999.0,
            "glob": "*__swe.tif"
          }
        },
        "ingester": "swann"
      }
    }
  },
  "pourpoint_index": "pourpoints/index.geojson",
  "pourpoint_records": "pourpoints/records",
  "sources": {
    "terrain": "/data/dem.tif"
  }
}
```

The referenced `data/snodas/dataset.json`, trimmed from the built-in
`swann-800m` template (`DATASET_TEMPLATES` in `snowtool.snowdb.datasets`,
which derives every template from a real built-in spec):

```json
{
  "resource": "snowtool.dataset/v1",
  "grid": {
    "origin_x": -125.0208,
    "origin_y": 49.9375,
    "px_size": 0.008333325394357,
    "cols": 7025,
    "rows": 3105,
    "tile_size": 256,
    "crs": 4269
  },
  "variables": {
    "swe": {
      "unit": {"name": "mm", "scale_factor": 1.0},
      "reducer": "mean",
      "dtype": "int16",
      "nodata": -999.0,
      "glob": "*__swe.tif"
    },
    "depth": {
      "unit": {"name": "mm", "scale_factor": 1.0},
      "reducer": "mean",
      "dtype": "int16",
      "nodata": -999.0,
      "glob": "*__depth.tif"
    }
  },
  "ingester": "swann",
  "zones": {
    "terrain": {"elevation": {"band_step_ft": 1000}},
    "landcover": {"forest_cover": {"threshold_pct": 50.0}}
  }
}
```

Rather than authoring one of these by hand, `dataset create --template
<kind>` stamps the built-in template for a dataset kind; hand-editing is
for tuning `zones` knobs, a `footprint`, or a `data_dir` afterward.

## How a config reaches runtime

Both the CLI and the API take a filesystem path to the snowdb: either the
`snowdb_conf.json` file itself or the directory containing it (`SnowDb.open`
appends the fixed filename when handed a directory). On the CLI the path
comes from `--config`/`-C` or the `SNOWTOOL_SNOWDB_CONFIG` environment
variable, resolved onto the per-invocation `CliContext` (`cli/_context.py`),
which lazily opens a single `SnowDb` the first time a command needs it — so
commands that never touch the database (`--version`, `api serve`) require no
config at all.

The API layer adds a small pydantic-settings model, `Settings`
(`api/settings.py`), with env prefix `snowtool_`. It exposes exactly three
knobs: `snowdb_config` (the `SNOWTOOL_SNOWDB_CONFIG` path, file or
directory), `tiff_cache_size` (the maximum number of open async-tiff
handles kept in the read-path LRU cache), and `max_zone_cells` (a cap on a
crossed zonal-stats query's product size — the number of output rows —
rejected before any raster is read, so several fine-grained zone axes
crossed together cannot blow up a query). There is no CORS or other web
knob here. Dotenv loading is deliberately disabled on this model: a `.env`
file is never parsed, so these variables must be set in the actual
environment (source them from a shell). The CLI depends on no `Settings` at
all — that is purely an API concern.
