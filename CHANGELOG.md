# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Removed

### Fixed

### Security

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

[Unreleased]: https://github.com/PSU-CSAR/snowtool/compare/v0.2.0...HEAD
[v0.2.0]: https://github.com/PSU-CSAR/snowtool/releases/tag/v0.2.0
[v0.1.0]: https://github.com/PSU-CSAR/snowtool/releases/tag/v0.1.0
