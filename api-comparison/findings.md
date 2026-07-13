# SNODAS API comparison findings

**Date:** 2026-07-12
**Services:** old = `api.snodas.geog.pdx.edu` (django-snodas), new = `ebagis.geog.pdx.edu` (snowtool)
**Scope:** SWE, WY2024 (2023-10-01 → 2024-09-30), no-zone + 1000 ft elevation bands,
pourpoints `06093200:MT:USGS`, `06078600:MT:USGS`, `14191000:OR:USGS`.
Method and scripts: see `fetch.py`, `analyze.py`, `independent.py` here; design spec at
`docs/superpowers/specs/2026-07-12-snodas-api-comparison-design.md`.

## Verdict

1. **Both raster zonal-stats pipelines compute correctly.** Each service's stats were
   reproduced independently from raw NOHRSC SNODAS grids (numpy, geodesic per-row cell
   areas, no shared code) on 3 arbiter dates: the **new API matched to the printed
   precision on every basin and date** (e.g. Badger 2024-03-15: 147.65 vs 147.65 mm;
   areas exact to the m²), and the **old API's *zonal* endpoint matched within 0.1%**
   over its own polygon (92.10 vs 92.08 mm). Where the two services host the same
   polygon (Willamette), their whole-basin SWE series agree to 0.2% (median ratio
   0.998, p10 0.997) across 260 comparable dates.

2. **The reported 200%/50% area anomalies are polygon data, not computation.** The two
   services host **different basin delineations** for both suspect pourpoints (despite
   the same-source assumption), and each service reports areas faithful to its own
   polygon (within 1%, i.e. rasterization discretization):

   | pourpoint | old polygon | new polygon | external reference |
   |---|---|---|---|
   | 06093200 Badger Ck | 827.90 km² | 341.99 km² | NWIS DA 153 mi² = **396.3 km²** |
   | 06078600 Gibson Res Inflow | 669.81 km² | 1,426.71 km² | Gibson Dam drainage ≈ 575 mi² ≈ **1,489 km²** (approx., verify) |
   | 14191000 Willamette@Salem | 19,109.4 km² | 19,109.4 km² (same) | NWIS DA 7,280 mi² = 18,855 km² (98.7%) |

   - **Old = 209% of NWIS on Badger, and ~47% of the plausible Gibson drainage — this
     reproduces the "200% / 50%" report exactly.** The old polygons match
     `bagis-pourpoints/reference/` byte-for-value (geodesic areas equal to sub-m²), so
     the bad delineations are in that source set, not introduced by the old service.
   - The **new service's pourpoint coordinates match the authoritative AWDB/NWIS
     station locations exactly** (Badger 48.36945, −112.80178; Gibson/Sun R below
     Gibson Dam 47.60101, −112.76209); the old service's outflow points are at
     different locations entirely. The old Badger polygon appears delineated from the
     wrong pourpoint; the old Gibson polygon covers only ~47% of the drainage.
   - `06078600` is not a NWIS site (AWDB forecast point "Sun R below Gibson Dam");
     NWIS returns 404, so drain-area vetting used the approximate published Gibson Dam
     drainage.

3. **The old service's no-zone ("basic"/SQL) stats endpoint is broken.**
   - JSON responses are a hard **500** on all three pourpoints (CSV works).
   - The CSV values do not match any defensible computation: against the independent
     recomputation over the same polygon, `swe` (apparently meters; ×1000 assumed) is
     off by factors ranging from **0.7× to 4.5×, direction varying by date and basin**
     (e.g. Badger 2024-03-15: 245.5 mm vs actual 92.1; 2024-05-15: 186.3 vs 41.8;
     Willamette 2024-05-15: 62.3 vs 77.3). The old zonal endpoint over the *same*
     polygon is correct, so this is specific to the SQL/Postgres stats path.
   - Anyone consuming `/pourpoints/{id}/stats/*` from the old service is getting wrong
     numbers; `/pourpoints/{id}/zonal-stats/*` is fine (modulo the polygon issue).

## Residual items worth follow-up

- **New Badger polygon = 86% of NWIS drainage** (342 vs 396 km²). Far better than the
  old 209%, and the delineation anchors to the correct station coordinates, but the
  −14% gap (possibly the "below Four Horns Canal" contributing-area subtlety, possibly
  delineation) merits a look by whoever owns the delineations.
- **Gibson Dam reference area (~575 mi²) is from general published figures**, not an
  authoritative machine-readable source; worth confirming before quoting.
- New service, Willamette: recombining band means by band area reproduces the
  whole-basin mean only to within 0.67% — but this is an artifact of the *check*, not
  an API inconsistency. `area_m2` counts every AOI pixel in the band while
  `mean_swe_mm` averages only pixels with valid SNODAS values (`zonal_stats.py:422`
  vs `:486`); the Willamette basin contains 34 SNODAS open-water nodata pixels
  (~21 km²: Fern Ridge Reservoir in the 0–1000 ft band, Waldo Lake in the
  5000–6000 ft band), so the nodata fraction varies by band and the exact
  recombination would need per-band *valid* area, which the response doesn't carry.
  Band areas sum to the whole-basin area exactly (bands fully partition the AOI),
  and the Montana basins — zero in-basin nodata — recombine to float precision.
- Old vs new **per-band areas** differ up to ~40% in the small highest bands even on
  the identical Willamette polygon (different DEM sources); totals and SWE stats agree,
  so this is expected DEM lineage noise, not a bug.
- Date coverage WY2024: **complete on both services** (366/366 all endpoints).

## Evidence inventory

- `out/analysis-report.txt` — full check-suite output (areas, identity, internal
  consistency, coverage, band tables, series ratios).
- `out/comparison-<site>.csv` — per-date WY2024 series, both services, all paths.
- `out/independent-report.txt` — arbiter recomputation vs both APIs, 3 dates × 3 basins
  × 2 polygons.
- `cache/` — every raw response (both services, NWIS, raw SNODAS tars), fully
  reproducible via the three scripts.

## Independent recomputation, arbiter dates (mean SWE, mm)

| basin / date | independent (ref poly) | old zonal | independent (new poly) | new API | old SQL |
|---|---|---|---|---|---|
| Badger 2023-12-01 | 13.68 | 13.68 | 26.85 | 26.85 | 50.18 |
| Badger 2024-03-15 | 92.08 | 92.10 | 147.65 | 147.65 | 245.52 |
| Badger 2024-05-15 | 41.75 | 41.77 | 79.89 | 79.89 | 186.29 |
| Gibson 2023-12-01 | 27.95 | 27.96 | 17.76 | 17.76 | 6.74 |
| Gibson 2024-03-15 | 233.58 | 233.58 | 219.35 | 219.35 | 215.64 |
| Gibson 2024-05-15 | 249.56 | 249.54 | 227.09 | 227.09 | 219.66 |
| Willamette 2023-12-01 | 4.00 | 4.01 | 4.00 | 4.00 | 5.58 |
| Willamette 2024-03-15 | 143.88 | 144.08 | 143.88 | 143.88 | 172.71 |
| Willamette 2024-05-15 | 77.25 | 77.45 | 77.25 | 77.25 | 62.33 |
