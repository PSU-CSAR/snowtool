# SNODAS API comparison: old (api.snodas.geog.pdx.edu) vs new (ebagis.geog.pdx.edu)

**Date:** 2026-07-12
**Status:** approved

## Goal

Determine whether either service returns incorrect SNODAS statistics — with a
specific eye on a secondhand report that pourpoint `06093200:MT:USGS` showed
~200% of its actual area and `06078600:MT:USGS` ~50% — and if a discrepancy
exists, localize it to a layer: polygon data, rasterization/mask, area
computation, zoning, or date handling. Neither service is presumed correct.

## Scope

- **Triplets:** `06093200:MT:USGS`, `06078600:MT:USGS`, `14191000:OR:USGS`
  (first two are the suspect basins; third is a control with no report).
- **Variable:** `swe` only (both services use this key).
- **Dates:** water year 2024 (2023-10-01 → 2024-09-30), adjusted after a
  coverage probe if either service has gaps.
- **Zones:** no-zone (whole basin) and elevation at 1000 ft bands — both
  services default to 1000 ft bands aligned to 0.

Both services were loaded from the same delineated basin polygons (confirmed
by the user), so geometry is nominally controlled; the comparison still
verifies this rather than assuming it.

## Non-goals

Root-causing a bug in either codebase (that is the follow-on, using the local
`rasterdb/` + django-snodas and a local snowdb, once this comparison says
which service and which layer is implicated). No reusable harness; this is a
one-off analysis.

## Shape

Untracked analysis directory `api-comparison/` at the repo root, holding
PEP-723 uv scripts (httpx, pyproj, numpy, rasterio as inline deps) plus a
`cache/` of raw responses so analysis reruns don't re-hit the services:

1. **`fetch.py`** — pulls and caches, per triplet from each service:
   pourpoint metadata + basin GeoJSON, the no-zone WY series, and the
   elevation-zoned WY series (JSON everywhere). Serial requests, resumable
   from cache. The old zonal endpoint computes from rasters per request, so
   the WY range is chunked (monthly) if a full-year call times out.
2. **`analyze.py`** — normalizes both response shapes into a common
   per-date, per-band table and runs the check suite, emitting per-pourpoint
   comparison CSVs and a check summary.
3. **`independent.py`** — the arbiter: for 3 dates (early accumulation,
   near-peak, melt), downloads raw masked SNODAS SWE from the NOHRSC archive,
   rasterizes the shared basin polygon onto the SNODAS 1/240° grid, and
   computes mean SWE and basin area with plain numpy using per-row geodesic
   cell areas — a third implementation sharing no code with either service.

Findings are written to `api-comparison/findings.md`.

## Check suite (dependency order)

1. **Polygon identity:** fetch the basin polygon from both services; verify
   geodesic areas match and coordinates are effectively identical. Failure
   here confounds everything downstream and means upstream data, not code.
2. **Geometry layer:** geodesic polygon area (`pyproj.Geod`) is the trusted
   area reference. Compare each API's whole-basin `area_m2` to it (expect
   ~1%, looser for small basins — ~926 m cells discretize the perimeter) and
   to NWIS `drain_area_va` as a coarse external flag for 2×/0.5× errors.
3. **Internal consistency, per service:** Σ(band areas) = whole-basin area;
   area-weighted mean of band means = whole-basin mean; old service only:
   SQL no-zone stats vs the zonal endpoint's whole-basin row (two independent
   implementations in one deployment).
4. **Cross-API series:** join on date; report one-sided dates; per-date
   mean-SWE deltas; whole-basin area ratio (constant? 2.0/0.5 on the suspect
   basins?); per-band area ratios (uniform vs elevation-dependent error).
5. **Independent recomputation:** on the 3 arbiter dates, the service that
   matches the numpy computation (mean within ~1%, area consistent with
   geodesic) is vetted; the other is implicated.

## Diagnosis map

| Observation | Suspected component |
|---|---|
| Means agree, areas scale by a constant | cell-area / area-raster computation |
| Means disagree too | AOI masks differ (rasterization) |
| Band sums ≠ whole-basin | zoning / crossing logic |
| Date sets differ | ingest / date handling |

## Error handling

Cached raw responses are the source of truth for analysis. Fetch failures
(409 no-polygon, 404, timeouts) are recorded per request and surfaced in the
report rather than aborting. Tolerances are reported as percentages, not
silent booleans, so borderline cases stay visible.
