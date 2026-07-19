# The stats surface: parameters and output formats

The [query engine](queries.md) computes a stats result — for a pourpoint, a
dataset, some variables and dates, and zero or more crossed zone axes it reduces
each date's rasters over the basin. This page is about the two ends wrapped
around that engine: **how a caller asks for a stratified query** (the zone
parameters, shared by the CLI and the HTTP API) and **how the result is
serialized** (the compact `json` body and the flat `csv` rows). The reduction
itself — area-weighting, the crossed index, nodata handling — lives in
[queries.md](queries.md); here it is the thing being asked for and rendered.

Both front ends sit on the same seam, `SnowDbReader.zonal_stats`
(`snowdb/reader.py`), and the same serializers on `ZonalStats`
(`snowdb/zonal_stats.py`): `dump_compact` for JSON and `dump_to_csv` for CSV.
There is exactly one JSON shape and one CSV shape, identical whether they come out
of the CLI or the API.

## At a glance

The same query — `snodas` at one date, stratified by elevation into two populated
1000 ft bands, reporting mean SWE — in both formats.

Compact JSON (the bare CLI body; over HTTP this is wrapped in an envelope):

```json
{
  "zone_layers": ["terrain.elevation"],
  "variables": ["mean_swe_mm"],
  "zones": [
    {"zone": [{"kind": "band", "layer": "terrain.elevation",
               "min": 6000, "max": 7000, "unit": "ft"}], "area_m2": 592891.69},
    {"zone": [{"kind": "band", "layer": "terrain.elevation",
               "min": 7000, "max": 8000, "unit": "ft"}], "area_m2": 811233.44}
  ],
  "results": {"2008-12-14": [[42.7], [51.3]]}
}
```

CSV — the same result, one row per (date, cell):

```csv
date,terrain.elevation_min_ft,terrain.elevation_max_ft,area_m2,mean_swe_mm
2008-12-14,6000,7000,592891.69,42.7
2008-12-14,7000,8000,811233.44,51.3
```

The correspondence: JSON's two `zones` are CSV's two rows, and
`results["2008-12-14"][i]` is the value row for `zones[i]` (here one variable, so
one number each). The sections below expand each format across multiple dates,
variables, and axis kinds.

## Asking for zones: the `LAYER:PARAM=VALUE` token

A query stratifies by zero or more **zone axes**. Each axis is a single string
token, and the CLI and API parse it with the *same* function,
`parse_zone_selection` (`snowdb/zonal_stats.py`):

```
LAYER                     whole default axis (e.g. terrain.elevation)
LAYER:PARAM=VALUE         override the axis' one scheme parameter
```

- `LAYER` is a zone-layer registry key, `'<provider>.<layer.key>'` —
  `terrain.elevation`, `terrain.aspect`, `landcover.forest_cover`.
- `:PARAM=VALUE` is optional and names the axis' single scheme parameter
  explicitly, so a token is self-describing rather than positional:
  `terrain.elevation:band_step_ft=500` (500 ft bands),
  `landcover.forest_cover:threshold_pct=40` (split at 40 %).
- `PARAM` must be exactly the parameter that layer's scheme advertises;
  a wrong name, a missing `=`, or an override on a categorical axis (which takes
  none) is a clean `QueryParameterError`. The value is typed and validated by the
  scheme itself (`ZoneScheme.parse_override`).

The [zones](zones.md) page owns the scheme kinds; their override parameters are:

| Scheme kind | Override `PARAM` | Meaning |
|-------------|------------------|---------|
| banded      | `band_step_ft`   | band width (feet) |
| bucketed    | `buckets`        | number of even buckets over the axis domain |
| threshold   | `threshold_pct` / `entropy_threshold` | the split point |
| categorical | *(none)*         | fixed classes; bare `LAYER` only |

**On the CLI** the axis is the repeatable `--zone` flag; **on the API** it is the
repeatable `zone` query parameter. They converge on the same
`list[ZoneSelection]` — only the input plumbing differs.

```bash
snowtool stats snodas 13120:CO:SNTL --dates 2024-01-01/2024-06-30 \
    --zone terrain.elevation:band_step_ft=500 --zone terrain.aspect
```
```
GET /datasets/snodas/stats/13120:CO:SNTL/date-range
    ?datetime=2024-01-01/2024-06-30
    &zone=terrain.elevation:band_step_ft=500
    &zone=terrain.aspect
```

### Discovery: the dataset resource is the form

The API endpoint's query schema is generic (one route family for every dataset),
so it cannot enumerate a dataset's valid zone keys or variables. That vocabulary
lives one hop away, on the dataset resource: `GET /datasets/{dataset}` returns
`DatasetInfo` (`api/models/dataset.py`), whose `zones[]` advertises each axis'
`key`, its override `param`, `default`, `unit`, and covered `min`/`max` (or
`classes` for a categorical axis), and whose `variables[]` advertises each
variable's `key`. Reading that resource tells you exactly which `LAYER`,
`PARAM`, and variable strings a query token may use — the request mirrors the
advertised structure. The dataset resource also carries templated `stats`
links (rels `stats-date-range` / `stats-doy`) a client can expand.

## From selection to cells

The engine crosses the K selected axes into a **mixed-radix product of cells**
(one `Zone` tuple per cell, in flat product order) — see
[queries.md](queries.md#zone-axes-and-the-crossed-index) for the radix math. Two
properties matter for the output:

- **Empty cells are dropped by default.** Crossing several fine axes makes most
  product combinations fall on no basin pixel (0 area). Those are omitted from
  both formats unless `--include-empty-zones` / `include_empty_zones=true` asks
  for the full product, in which case they appear with `area_m2` 0 and `null`
  values.
- **`area_m2` is a property of the cell, not the date.** It is the cell's total
  in-basin geographic area, computed once from the AOI raster and identical for
  every date — so the JSON format states it once per cell, and a cell can report
  real area with a `null` value (it covers ground, but no data exists there for
  that variable and date).

`K = 0` (no `--zone`) is the whole-basin case: a single cell whose zone tuple is
empty.

## The compact JSON format

`ZonalStats.dump_compact` produces `CompactStats` (`snowdb/zonal_stat_models.py`)
— a **normalized** body that defines zones and variables once and reduces each
date to a bare numeric matrix:

```jsonc
{
  "zone_layers": ["terrain.elevation"],          // the crossed axes, in selection order
  "variables":   ["mean_swe_mm", "total_swe_m3"],// stat names, = inner-array order
  "zones": [                                      // one per emitted cell, flat product order
    {"zone": [{"kind": "band", "layer": "terrain.elevation",
               "min": 6000, "max": 7000, "unit": "ft"}],
     "area_m2": 592891.69},
    {"zone": [{"kind": "band", "layer": "terrain.elevation",
               "min": 7000, "max": 8000, "unit": "ft"}],
     "area_m2": 811233.44}
  ],
  "results": {                                    // date -> matrix[zone_idx][variable_idx]
    "2008-12-14": [[42.7, 25318.1], [51.3, 41627.0]],
    "2008-12-15": [[43.1, 25550.0], [null,  null]]
  }
}
```

- `zones` and `variables` are stated **once**; each date in `results` is just a
  `zones × variables` grid of numbers. The outer index aligns to `zones`, the
  inner to `variables`. This is the whole point of the format: the per-date cost
  is numbers only, so a multi-year daily query does not re-emit the zone/variable
  structure thousands of times.
- `null` (via the `StatValue` `nan → None` validator) means a variable with no
  valid pixels that date, or an empty zone.
- `area_m2` is **hoisted** into each `zones[]` entry (date-invariant, above).
- The body carries no per-dataset field names (variables are strings, values are
  positional), so it is **one schema for every dataset**.

Each `zones[].zone` entry is a self-describing per-axis ref, discriminated on
`kind` (`ZoneRef` in `snowdb/zonal_stat_models.py`):

| `kind`      | fields |
|-------------|--------|
| `band`      | `layer`, `min`, `max`, `unit` |
| `class`     | `layer`, `code`, `label` |
| `threshold` | `layer`, `threshold`, `unit`, `side` (`below`/`above`), `label` |

A whole-basin (`K = 0`) result has `zone_layers: []` and a single zone whose
`zone` is `[]`. A query that selects no dates has `results: {}`.

**CLI vs. API envelope.** The CLI prints this body bare. The API wraps it in
`CompactStatsResponse` (`api/models/stats.py`) — the same body plus the
`pourpoint`, `dataset`, echoed `query`, and HATEOAS `links` (including the `csv`
alternate). The CLI minifies the JSON when stdout is piped and pretty-prints
(`indent=2`) only when it is a TTY (the `jq`/`git` convention); the API always
emits minified JSON.

## The CSV format

`ZonalStats.dump_to_csv` is the flat, row-oriented twin: **one row per (date,
cell)**, drawing from the same `_zone_stats` source as the JSON path, so the two
never disagree on a value. Each zone axis names its own columns via
`Zone.csv_columns` (`snowdb/zones/zoning.py`):

| axis kind   | columns |
|-------------|---------|
| banded      | `{layer}_min_{unit}`, `{layer}_max_{unit}` |
| categorical | `{layer}` (the class label) |
| threshold   | `{layer}_side`, `{layer}_threshold_{unit}` |

The header is `date`, then each selected axis' columns (in selection order), then
`area_m2`, then one column per variable `stat_name`. A `null`/no-data value is an
empty cell, never the literal `nan`. For a single `terrain.elevation` axis over
one SWE variable:

```csv
date,terrain.elevation_min_ft,terrain.elevation_max_ft,area_m2,mean_swe_mm
2008-12-14,6000,7000,592891.69,42.7
2008-12-14,7000,8000,811233.44,51.3
```

Empty-cell dropping and `include_empty_zones` work exactly as for JSON. Over
HTTP the download filename comes from the query (`DateQuery.csv_name`): the
pourpoint, the date span, and a `_zonal_<n>` suffix when zones are crossed.

## Choosing a format

- **API:** content negotiation. `?f=json` (the default) or `?f=csv`, or an
  `Accept: application/json` / `Accept: text/csv` header —
  `_StatsFormat`/`negotiate` in `api/routers/stats.py`. The JSON envelope
  advertises the CSV as an `alternate` link.
- **CLI:** `--format json` (the default) or `--format csv` on the `stats`
  command.

Both endpoints and both formats are the *same* result rendered two ways; the
choice is purely presentational.

## Where to read next

- [The query engine](queries.md) — the reduction, the crossed index, reducers,
  date selection, and the read path that produce the numbers rendered here.
- [Zone layers](zones.md) — the schemes behind the axes and their `assign`/
  override contracts.
- [Pourpoints and AOI rasters](pourpoints.md) — the AOI raster that is both the
  basin mask and the area weights.
