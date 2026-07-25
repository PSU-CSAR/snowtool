# Checking a snowdb: `doctor`

`snowtool doctor` is a **read-only health inspection** of a snowdb. It never
changes anything on disk — it opens artifacts, compares them against what the
config declares, and reports what disagrees. Empty output means healthy; any
finding makes the command exit non-zero, so it works as a cron/CI gate:

```console
snowtool doctor                       # all checks, all active datasets
snowtool doctor grid pourpoints       # only these checks
snowtool doctor -d snodas             # only this dataset (resolves even if inactive)
snowtool doctor --format json         # machine-readable rows
snowtool doctor --include-inactive    # also check registered-but-inactive datasets
```

Each run prints what it is about to check, then a progress bar with one
increment per unit of work (per COG, per pourpoint basin, per AOI raster, per
date). On a large store the enumeration takes a moment; the startup line tells
you it is working.

## What `doctor` is (and is not) for

`doctor` **detects inconsistencies** between the config and the on-disk reality,
and structural/alignment/integrity problems it can verify cheaply by reading.
It does **not** enforce freshness against upstream sources, and it never
rebuilds anything. Fixing is the job of the converge commands:

| To fix… | Run |
| --- | --- |
| Missing or stale COGs | `snowtool dataset ingest …` |
| Missing or stale AOI rasters | `snowtool pourpoint rasterize [--rebuild]` |
| Missing or stale zone layers | `snowtool dataset generate-zones …` |
| A stale coverage index | `snowtool pourpoint reindex` |

So the usual loop is: `doctor` to see what's wrong → the matching converge
command to fix it → `doctor` again to confirm it's clean.

## Output

Every finding is one row with four fields:

| Field | Meaning |
| --- | --- |
| `check` | which check produced it: `grid`, `dates`, `files`, or `pourpoints` |
| `dataset` | the dataset the finding is about |
| `target` | what within the dataset — a file, a date, a station triplet, an artifact name |
| `issue` | a short description of the problem |

When one target trips several issues, they are joined with `; ` onto a single
row rather than repeated.

## The checks

- **`grid`** — declaration-vs-reality. Flags an ingester that declares no
  variables, and validates the shape + transform of every full-grid artifact
  (each COG, each zone-layer file, and the nodata mask) against the declared
  grid.
- **`dates`** — every ingested date is complete: each of the dataset's variables
  resolves to exactly one COG for that date.
- **`files`** — the dataset's expected artifacts exist (COGs, AOI rasters, zone
  layers), and built zone-layer sets carry the current on-disk format version.
- **`pourpoints`** — how the stored pourpoints line up with the dataset's grid
  and burned AOI rasters: geometric coverage, missing/orphan rasters, and the
  health of each burned AOI raster (readable, aligned, and current).

## Issues and what to do about them

### `grid`

| Issue (`target`) | Meaning | Fix |
| --- | --- | --- |
| `has an ingester but declares no variables` (empty) | A misconfigured dataset spec — an ingester with nothing to write. | Fix the dataset definition. |
| `declared grid is CxR (cols x rows) but artifact is cxr` (a file) | The file's pixel dimensions don't match the declared grid. | If the *config* is right, rebuild the file (re-ingest / regenerate). If the *file* is right, the declared grid drifted — reconcile it. |
| `declared grid transform (…) does not match artifact transform (…)` (a file) | The file sits on a different lattice (origin/resolution) than the declared grid. | Same as above. A common cause is changing a dataset's declared grid: re-ingest COGs, `generate-zones` for zone layers, and re-stamp the nodata mask onto the new grid. |

### `dates`

| Issue (`target` = a date) | Meaning | Fix |
| --- | --- | --- |
| `missing VAR, VAR, …` | That date is ingested but incomplete — those variables have no COG (or a duplicate COG makes them ambiguous). | Re-ingest the source for that date; remove any duplicate COG. |

### `files`

| Issue (`target`) | Meaning | Fix |
| --- | --- | --- |
| `missing` (`cogs`) | No COGs ingested yet. | `dataset ingest`. |
| `missing` (`aoi-rasters`) | No AOI rasters burned. | `pourpoint rasterize`. |
| `missing` (`terrain (elevation.tif, …)`) | A zone-layer set is absent or incomplete; the named files are missing. | `dataset generate-zones`. |
| `stale zone-layer format (stored X != current Y)` (a provider) | The built set's on-disk format version is older than the code's. | `dataset generate-zones` to rebuild. |

### `pourpoints`

| Issue (`target` = a station triplet) | Meaning | Fix |
| --- | --- | --- |
| `no coverage` | The basin is entirely outside this dataset's grid. Often expected (e.g. an Alaska station on a CONUS-only dataset). | Usually none. Investigate only if the pourpoint *should* be covered. |
| `partial coverage` | The basin overlaps the grid but spills outside it; a query uses only the in-grid portion. | Usually informational. Pass `allow_partial` when querying if that's acceptable. |
| `no raster` | A *covered* basin has no burned AOI raster. | `pourpoint rasterize`. |
| `orphan raster` | A burned AOI raster exists with no backing pourpoint record. | Remove the stray raster, or restore the record. |
| `empty AOI raster (covers no in-grid cells: off-grid or masked)` | The burned raster has no in-basin cells — see below. | Depends on cause (below). |
| `missing SNOWTOOL_TILE_BBOX tag (rebuild with pourpoint rasterize --rebuild)` | A legacy/corrupt AOI raster without its window metadata. | `pourpoint rasterize --rebuild`. |
| `unreadable: …` | The AOI raster file could not be opened. | `pourpoint rasterize --rebuild`. |
| `stale content (stored … != expected …)` | The basin changed since the AOI raster was burned (or the nodata mask changed). | `pourpoint rasterize` — the converge rebuilds it. |
| `declared grid transform (…) does not match artifact transform (…)` | The AOI raster was burned on a grid that has since moved. | `pourpoint rasterize` — a grid move now marks it stale, so the converge rebuilds it. |

## `no coverage` vs. `empty AOI raster`

These look similar but come from different checks and mean different things:

- **`no coverage`** is geometric: the basin polygon does not intersect the
  dataset's grid at all. Computed live from the stored basin, independent of any
  raster.
- **`empty AOI raster`** is about a burned file: an AOI raster exists on disk but
  is all-zero, so it contributes no pixels to any query.

An off-grid basin cannot be rasterized, so a *missing* raster there is expected
and `no raster` is suppressed — only `no coverage` is reported. But an all-zero
raster on disk for an off-grid basin is a **stray artifact** the normal flow
never writes, so it is flagged (`empty AOI raster`) alongside `no coverage`.

Note that **coverage does not fully predict emptiness.** A basin that *is*
covered can still burn all-zero when every cell it overlaps is removed by the
dataset's nodata mask (e.g. a small basin entirely over open water). Coverage is
a pure geometry test and doesn't consult the mask, so `empty AOI raster` is the
only signal for that case — which is why the check reads the burned array rather
than inferring emptiness from coverage.

## Under the hood

The same checks power both `doctor` and the write path: `pourpoint rasterize`
skips a raster only when the shared AOI-raster check reports no *actionable*
issue, and rebuilds otherwise (a grid move, a changed basin, a corrupt file, or
a missing tag all force a rebuild; `--rebuild` forces it regardless). Findings
are typed objects internally, so `doctor`'s report and the writer's skip decision
can never drift apart. See
[Provenance and staleness](internals/provenance.md) for how the underlying
hashes and format versions work.
