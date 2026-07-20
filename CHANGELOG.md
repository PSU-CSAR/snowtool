# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A compact, normalized zonal-stats representation — zones and variables are
  defined once and each date's values are a bare `zones × variables` matrix
  (`null` for a zone with no valid pixels), with `area_m2` hoisted into the zone
  definition — with one response schema for every dataset. This is now the `json`
  output of both `snowtool stats` and the HTTP stats endpoint (see **Changed** for
  the endpoint unification and the retired verbose form). The dataset resource
  (`GET /datasets/{dataset}`) advertises the valid zone keys, override params, and
  variables for building a query.
- **CLI change:** the `stats --zone` override syntax is now the explicit
  `LAYER:PARAM=VALUE` (e.g. `terrain.elevation:band_step_ft=500`); the old
  positional `LAYER:VALUE` form is no longer accepted.

### Changed

- **Breaking:** `windows iis install`'s `--skip-site`/`--skip-config` flags
  are replaced by a single `--only [config|site]` selector (default: run
  both steps).
- **Breaking (stats output):** `json` zonal-stats output is now the compact,
  normalized body (zones and variables defined once; each date a
  `zones × variables` matrix; `area_m2` hoisted; `null` for empty zones). The
  HTTP stats API is now a single generic endpoint,
  `GET /datasets/{dataset}/stats/{triplet}/{date-range,doy}`, with `{dataset}` a
  path parameter; CSV is still available via `?f=csv` / `Accept: text/csv`. The
  CLI `stats` command now defaults to `--format json` and prints minified JSON
  unless stdout is a TTY (then it pretty-prints).
- **Breaking (error codes):** on the HTTP stats endpoint, an unknown/invalid
  `zone` or `variable` value is now a `422` (raised by the shared zone-selection
  parser) rather than the previous `400` (schema-level enum rejection on the
  old per-dataset query model).
- **Breaking (error codes):** on the day-of-year stats endpoint, an
  impossible calendar day (e.g. Feb 30) or an inverted year span is now a
  `400` `/problems/malformed-query-parameter` (FastAPI-native request
  validation) rather than the previous `422` (round-tripped through
  `QueryParameterError`). This moves the opposite direction from the
  zone/variable change above because it's a different class of parameter
  error: malformed request-scope input, not an unresolvable domain value.
- **Breaking (`dataset info` output):** the `json`/`csv` forms are now typed
  and machine-stable: `cell_area_m2` is `null` (not the prose string
  `"varies (geographic)"`) on a geographic grid, and the elevation bracket is
  the two numeric fields `min_elevation_m`/`max_elevation_m` (not the prose
  `elevation_bracket_m` string). The `table` form is unchanged.
- **Internal API:** `Dataset.rasterize_aoi` and `Dataset.rasterize_aoi_if_needed`
  are merged into a single `Dataset.rasterize_aoi(pourpoint, *, rebuild=False)`,
  converge-by-default (build when missing/stale, skip when current unless
  `rebuild=True`), returning the built `AOIRaster` or `None` when skipped as
  current. The redundant, differently-behaved `SnowDbManager.rasterize_aoi` is
  deleted (production always went through `rasterize_aois`); use
  `SnowDbManager.rasterize_aois` or the `Dataset` method directly.
  `Dataset.create`'s `force` parameter is renamed `exist_ok` and made
  keyword-only, matching what it actually does (`mkdir(exist_ok=...)`).

### Removed

- The verbose per-date zonal-stats JSON representation and its per-dataset
  response schemas, the CLI `--format json-compact` value (folded into `json`),
  and the per-dataset stats query-parameter compiler. Zone selection is the
  `LAYER:PARAM=VALUE` token grammar everywhere.

### Fixed

### Security

## [v0.3.0] - 2026-07-13

### Added

- Datasets can declare an optional `nodata_mask` in their config: a
  single-band raster on the dataset's grid whose 0/nodata pixels can never
  report data (e.g. SNODAS open water). Masked pixels are burned out of AOI
  rasters (zero area weight), so stats `area_m2` counts only pixels the
  dataset can actually report and per-band means recombine exactly to
  whole-basin means. The mask file's hash rides in AOI provenance
  (`SNOWTOOL_AOI_HASH`), so adding, changing, or removing a mask marks the
  dataset's AOI rasters stale — run `snowtool pourpoint rasterize` to
  converge. **Live databases:** configuring a mask changes reported basin
  areas for basins containing out-of-domain pixels (they shrink by the
  masked area); maskless datasets are unaffected (no rebuild).
- The SNODAS template ships its fixed water/off-domain mask (139 KB, derived
  from the SWE nodata footprint) as package data; `dataset create --template
  snodas` stamps it into the new dataset automatically.
- The SWANN 800m template ships its fixed off-domain mask (85 KB, derived from
  the SWE `!= -999` footprint — the CONUS land domain is ~54% of the grid
  rectangle, the rest permanent nodata) as package data; `dataset create
  --template swann-800m` stamps it in automatically, exactly as SNODAS does.
  **Live databases:** re-run `dataset create --template swann-800m <name>` on
  an existing SWANN dataset to converge — `create` is idempotent: it overwrites
  the mask, preserves the registration and active state, and re-burns only the
  AOI rasters the added mask marks stale. Basins with out-of-domain pixels then
  report the smaller in-domain area. INSTARR needs
  no such mask: its nodata is per-date/per-variable (cloud and snow-property
  gaps), not a fixed domain, and its one static gap (the empty MODIS corner)
  is already handled by the dataset `footprint`.

### Changed

### Removed

### Fixed

### Security

## [v0.2.2] - 2026-07-12

### Added

- The rendered `web.config` pins `GDAL_DATA`/`PROJ_DATA`/`PROJ_LIB` to the
  rasterio wheel's bundled data directories, so the hosted process is
  immune to ambient GDAL/PROJ environment variables (PostGIS, ArcGIS, and
  QGIS installs commonly set them system-wide, and IIS worker processes
  inherit them) pointing it at incompatible data from another installation.

## [v0.2.1] - 2026-07-12

### Added

- `windows iis remove` strips the app-pool account's permission grants from
  the venv, its base interpreter, the snowdb directory, and the site
  directory, making it a proper inverse of `install` (the grants name the
  pool's virtual account, whose SID derives from the pool name alone — left
  behind, they would silently re-attach to any future same-named pool).
- `windows iis install` grants the app-pool identity read+execute on the
  uv-managed Python interpreter backing the tool venv (`sys.base_prefix`);
  without it the site's child process dies at startup with "Access is
  denied" whenever the interpreter lives in the installing user's profile.

### Changed

- **Breaking:** `windows iis remove` requires `--config` (or
  `SNOWTOOL_SNOWDB_CONFIG`) to locate the snowdb grant it removes.
- The install-for-all-users instructions (docs and the `add-to-path`
  guidance message) also set `UV_PYTHON_INSTALL_DIR`, keeping the
  interpreter backing the tool venv out of the installing user's profile,
  and `UV_LINK_MODE=copy`, since hardlinks into uv's per-user cache carry
  the cache's user-only ACLs regardless of where they are linked.
- The IIS docs require IISAdministration ≥ 1.1.0.0 (Windows Server 2016's
  inbox 1.0.0.0 lacks `New-IISSite -Protocol`/`-CertificateThumbPrint`).
- The IIS permission grants apply the inheritable ACE at each tree root
  instead of rewriting every file's ACL (`icacls` without `/T`).

### Fixed

- `windows iis install` and `remove` work end-to-end; the provisioning
  scripts were broken since their introduction:
  - App-pool creation/removal use `New-WebAppPool`/`Remove-WebAppPool`
    (WebAdministration, now imported explicitly) — the previously called
    `New-IISAppPool`/`Remove-IISAppPool` do not exist in any module.
  - Re-runs converge instead of throwing duplicate-collection-entry errors:
    pool creation is existence-guarded and the site is dropped and
    recreated (`-Force` makes neither `New-WebAppPool` nor `New-IISSite`
    idempotent).
  - An https site without `--cert-thumbprint` is created through the raw
    ServerManager API, since `New-IISSite` refuses a certificate-less https
    binding; the certificate is then bound manually afterward, as already
    documented.
  - The shared ServerManager is refreshed after the WebAdministration pool
    writes and before site-level settings, fixing "file has changed on
    disk" commit failures on fresh installs and a NullReferenceException
    when re-binding the site's app pool.
  - Existence probes no longer leak spurious "does not exist" warnings.

## [v0.2.0] - 2026-07-12

### Added

- Top-level `doctor` command running the health checks (grid, dates, files,
  pourpoints) with a non-zero exit on any finding; replaces `snowdb validate`
  and the findings reports.
- Top-level `stats` command with OGC-interval `--dates`/`--years` date
  selection, sharing the API's datetime semantics.
- `dataset dates --missing` lists the absent dates within a range.
- `dataset register`/`create`/`remove-date` lifecycle: `create` stages and
  registers a dataset (inactive) from a built-in template; destructive
  operations (`dataset remove-date`, `pourpoint remove`) prompt for
  confirmation and support `--yes` and `--dry-run`.
- Machine-readable `--format json` results for `dataset ingest` and
  `pourpoint rasterize`.
- Rich console output: tables, progress bars and spinners (including ingest
  progress), and root `--color`/`--quiet` options with matching `SNOWTOOL_*`
  environment variables.
- API: stats query parameters are individually typed and documented per
  dataset (a per-dataset `variable` enum, per-axis zone and override
  parameters, `allow_partial`); dataset resources advertise templated stats
  links, the served zone layers with their value ranges, and the served
  dataset names as an enum; malformed-query-parameter 400s carry a
  resolvable problem type.
- API: a pourpoint's detail response advertises per-dataset templated stats
  links (one date-range + day-of-year pair per active dataset covering the
  basin), with the triplet bound into the path and a machine-readable
  `dataset` key on each link.
- Dataset-config `dtype` and grid `crs` values are validated at config load
  (numpy/pyproj must parse them) instead of failing at first raster read or
  grid build.

### Changed

- **Breaking:** the CLI command tree was reorganized — `init` and `status`
  are top-level commands (the `snowdb` group is gone), and the dataset read
  surface (`dates`, `info`, `values`, `list`) lives in the `dataset` group.
- **Breaking:** operator-facing errors are typed `SnowtoolError`s that the
  CLI maps centrally to messages and exit codes.
- **Breaking:** dataset-config zone params parse to per-scheme models
  (`band_step_ft`, `buckets`, `threshold_pct`, `entropy_threshold` are mutually
  exclusive per layer); a param attached to a layer whose scheme does not take
  it is now a `ZoneParamsError` instead of being silently ignored, and a layer
  enabled with no params is stored as `null` rather than `{}`.
- **Breaking (API):** a dataset's `zones` entries are a discriminated union on
  `kind`; fields that don't apply to a kind (e.g. `classes` on a banded axis,
  `param` on a categorical one) are now absent instead of `null`.
- Pourpoint sources parse through typed GeoJSON envelope models: the station
  triplet `id` is pattern-validated at import, a `Feature` source's geometry
  must be a `Point`, a `GeometryCollection` must be exactly point + basin, and
  a source with `null`/missing `properties` is classified invalid instead of
  crashing the import batch.
- Empty (zero-area) crossed zones are dropped from stats output by default;
  `--include-empty-zones` restores them.
- The aspect components (northness/eastness) are bucketed by count instead
  of a percent band width; the aspect-entropy unit is labeled `Hnorm`.
- The `windows` command group is hidden on non-Windows platforms.

### Removed

- The dataset `prune` command; use the confirmation-gated
  `dataset remove-date`.

### Fixed

- SNODAS snowpack average temperature is scaled from tenths of a kelvin.
- Tables are never wrapped on non-TTY output.

## [v0.1.0] - 2026-07-06

Initial release 🎉

[Unreleased]: https://github.com/PSU-CSAR/snowtool/compare/v0.3.0...HEAD
[v0.3.0]: https://github.com/PSU-CSAR/snowtool/compare/v0.2.2...v0.3.0
[v0.2.2]: https://github.com/PSU-CSAR/snowtool/compare/v0.2.1...v0.2.2
[v0.2.1]: https://github.com/PSU-CSAR/snowtool/compare/v0.2.0...v0.2.1
[v0.2.0]: https://github.com/PSU-CSAR/snowtool/releases/tag/v0.2.0
[v0.1.0]: https://github.com/PSU-CSAR/snowtool/releases/tag/v0.1.0
