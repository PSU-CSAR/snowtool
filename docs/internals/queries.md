# The query engine

A query is the one thing a snowdb exists to answer: pick a **pourpoint**, a
**dataset**, some **variables**, a set of **dates**, and zero or more **zone
axes**, and the engine reduces each date's per-variable raster over the basin,
stratified by the crossed zones. The whole computation lands in one method,
`SnowDbReader.zonal_stats` (`snowdb/reader.py`) — the shared read seam behind
both the CLI `query stats` command and the API's per-dataset stats endpoints
(see the [architecture](architecture.md) overview). It guards coverage, loads
the burned AOI raster, resolves the requested variables, builds the raster
collection for the query's dates, and hands all of it to
`ZonalStats.calculate` (`snowdb/zonal_stats.py`), which is the engine proper.

## The AOI raster is both membership and weights

The reduction never sees the basin polygon. What it sees is the **AOI raster**
(`snowdb/aoi_raster.py`): the basin burned onto the dataset grid as per-pixel
geographic cell area in m² *inside* the basin and `0` outside. That single
raster does double duty — `array > 0` is the in/out-of-basin membership signal,
and the same values are the area weights the reduction needs, with no separate
area raster. Its construction, tiling, and provenance are the subject of the
[pourpoints](pourpoints.md) page; here it is just an input.

Because the weights are area, both reducers are area-weighted.
`Reducer.MEAN` (`snowdb/variables.py`) is the area-weighted average
`Σ(value·area) / Σ(area)` over the pixels that actually carry data, and
`Reducer.TOTAL` is the area-weighted accumulation `Σ(value·area)` — a basin
total for an extensive quantity like a volume, *not* a bare `Σ(value)`. On a
projected grid every cell is equal-area and MEAN degenerates to a plain mean; on
a geographic grid the geodesic per-row area does the weighting for free.

Nodata in a data band is handled per pixel, not per cell. The reduction runs
only over the selection `in_zone & (values != variable.nodata)` — so a fill
pixel is dropped from both the numerator and the MEAN denominator. (This is why
`DatasetVariable.nodata` must be a finite sentinel: the mask is a `!=` compare,
and `x != NaN` is always true, so a NaN fill would never be excluded.) The
`area_m2` an output cell reports is separate from this: it is the cell's total
geographic area over every in-zone pixel, computed once from the AOI raster and
independent of the reducer or of which pixels happen to be nodata. A cell can
therefore report a real `area_m2` and a null value at once — it covers ground,
but no data exists there for this variable and date.

## Zone axes and the crossed index

Each stratification axis is a `ZoneSelection`: a zone-layer registry key plus an
optional scheme override, parsed from a `LAYER[:override]` token by
`parse_zone_selection`. The key is `'<provider>.<layer.key>'` —
`terrain.elevation`, `terrain.aspect`, `landcover.forest_cover` — and the
optional override is the axis' one scheme parameter (a band step for a banded
layer, a split threshold for a threshold layer; a categorical axis takes none,
and a token for one is a clean error). An empty selection means no
stratification at all: a single whole-basin cell per date.

The zone layers themselves are read *live* at query time — each selected layer
is opened, windowed to the AOI, and its pixels assigned to per-axis ordinals by
its `ZoneScheme.assign` (`snowdb/zones/zoning.py`), which returns `-1` for any
pixel that is layer-nodata or out of the scheme's domain. Nothing about the
zones is baked into the AOI raster, so a terrain or land-cover rebuild changes a
query's zones with no re-rasterization of the basin (see
[provenance](provenance.md) for why that decoupling holds). The
[zones](zones.md) page covers the schemes and their `assign` contracts in depth.

The K selected axes are crossed into a **mixed-radix product index**,
`_ZoneIndex` (`snowdb/zonal_stats.py`). Given K per-axis ordinal arrays, it
folds them into one flat cell index per pixel with the radix recurrence
`combined = combined * dim + ords`, iterating the axes in selection order (the
first axis is the outermost, most-significant digit). A pixel is **in-zone**
only
where the AOI area is positive *and* every axis assigned it a real ordinal
(`>= 0`); a `-1` on any single axis drops the pixel from every crossed cell. The
per-cell area is then a `bincount` of the AOI area over the in-zone pixels,
keyed by the combined index, so all cells' areas fall out of one pass. `K = 0`
is the whole-basin case: a single cell whose zone tuple is empty and whose area
is the whole in-basin area. `K = 1` is a single-axis stratification.

A worked example makes the radix concrete. Cross a 3-band elevation axis with a
2-side forest axis: `dims = [3, 2]`, so `prod = 6` cells. A pixel in elevation
band 2 (ordinal `2`) that is forested (ordinal `1`) lands in cell
`2 * 2 + 1 = 5`; band 0 unforested is cell `0`; band 1 forested is
`1 * 2 + 1 = 3`. The six cells enumerate in that flat order —
`(band0, below), (band0, above), (band1, below), (band1, above), (band2, below),
(band2, above)` — each carrying its per-axis `Zone` tuple, which is what the
result reports.

## The guard rail

Because every axis' zone count is known from its scheme *before any raster is
read*, the engine multiplies them and rejects a runaway crossing up front: a
product exceeding `max_zone_cells` raises `QueryParameterError` with no I/O. The
library default is `DEFAULT_MAX_ZONE_CELLS = 10_000`. The HTTP layer can raise
or lower it — `Settings.max_zone_cells` (`api/settings.py`, env
`SNOWTOOL_MAX_ZONE_CELLS`) is threaded into `SnowDbReader.max_zone_cells` —
while the CLI and tests take the default.

## Selecting dates

The temporal side is a small discriminated union, `PourPointQuery`
(`snowdb/query.py`), of two shapes. `DateRangeQuery` is an inclusive
`[start, end]` interval where either bound may be open (`None`) — it is a
*filter* over the dates a dataset actually has, so an open end simply drops that
side's constraint (OGC `datetime` interval semantics). `DOYQuery` is a
day-of-year query: a fixed `month`/`day` selected across every year in
`[start_year, end_year]`, for pulling e.g. every April 1 across two decades. It
validates that the month/day can occur (rejecting Feb 30) and that the year span
is not inverted, so a typo surfaces as an error rather than silently matching
nothing. Both satisfy the `DateQuery` protocol: `select(available)` returns the
matching subset of the dataset's ingested dates, and `csv_name(...)` builds the
download filename (the pourpoint, the date span, and a `_zonal_<n>` suffix when
zones are crossed).

## The read path

For the selected dates and variables, `RasterCollection`
(`snowdb/raster/collection.py`) resolves one `DataRaster` per (variable, date).
It validates completeness before any read: a date present for some requested
variables but not others is a partial/crashed ingest, surfaced as a typed
`IncompleteDatasetDataError` naming the missing variable rather than a silent
gap in the output.

Reads are windowed to the AOI's tile bounding box. The AOI raster records a
`SNOWTOOL_TILE_BBOX` tag, and only the tiles in that box are ever fetched — the
per-pixel mask nulls everything outside the basin, so no full-grid read happens
(see [pourpoints](pourpoints.md)). Every layer and data band is loaded through
`AOIRaster.load_raster_tiles_into_array`, which coalesces one COG's blocks into
a single batched fetch and places them into the window array. The engine fans
out across the whole job with `asyncio.gather` — the selected zone layers
concurrently, then every (variable, date) raster concurrently — over a shared
`TiffCache` that bounds and dedupes open COG handles (detailed under the
[raster read path](architecture.md#the-raster-read-path)).

## Output shapes

Internally a computed `ZonalStats` is a dense `float64` array indexed by
`(date, cell, stat)`, where stat `0` is `area_m2` and the rest are the reduced
variables. An empty cell — no selected pixels — is `nan`, set explicitly so a
no-data TOTAL reads null rather than a spurious `0`. Two serializers share that
array.

`dump()` builds the JSON structure from the per-dataset models generated in
`snowdb/zonal_stat_models.py`. Each date is one `…ZonalStats` object carrying
the `date`, the echoed `zone_layers`, and a flat list of `…ZonalStat` cells. A
cell
is self-describing: a `zone` array of one `ZoneRef` per crossed axis
(`BandZoneRef`, `ClassZoneRef`, or `ThresholdZoneRef`, discriminated on `kind`),
plus `area_m2` and one field per variable named `<reducer>_<key>_<unit>` (e.g.
`mean_swe_mm`). The variable fields are typed `StatValue`, which normalizes a
no-valid-pixels `nan` to JSON `null` at construction, so the payload is always
valid JSON. A whole-basin query is a cell with an empty `zone` array; crossing
more axes lengthens each cell's `zone` without changing the schema, and the list
flattens 1:1 to CSV.

`dump_to_csv()` writes one row per (date, cell). Each axis describes its own
columns: a banded or threshold axis expands to two typed, unit-bearing columns,
a categorical axis to one; then `area_m2` and the variable stats. A no-data cell
renders as an empty CSV field, never the literal `nan`.

At the CLI, `query stats --format json` emits the `dump()` models and the CSV
format streams `dump_to_csv`. The API wraps `dump()` in a `StatsResponse`
envelope (`api/models/stats.py`) — the pourpoint, the echoed query, HATEOAS
links, and the `results` list — or streams the same CSV, chosen by content
negotiation. The exact endpoints are the
[HTTP API reference](../reference/http-api.html)'s to document.

## Edge cases worth knowing

Three near-empty situations behave distinctly. A zone cell with **no pixels at
all** (an elevation band the basin never reaches, or a crossing excluded on some
axis) reports `area_m2 == 0` and a null value. A cell that **covers ground but
whose data band is entirely nodata** there reports a real `area_m2` and a null
value — the area counts the pixels, the reduction finds nothing to reduce. And a
**date missing a requested variable's file** never reaches the reduction: the
`RasterCollection` completeness check fails first, naming the missing variable,
so a partial ingest is an error rather than a hole in the results.
