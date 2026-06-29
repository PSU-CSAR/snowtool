# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

`uv` manages the venv and deps; `pytest`/`ruff`/`mypy` run inside it.

```bash
uv sync --dev                      # install (incl. dev deps)
pytest                             # all tests; network tests deselected by default
pytest tests/snowdb/test_db.py::test_open_requires_config   # single test
pytest -m network                  # opt-in: real S3 reads (3DEP, NLCD)
ruff check --fix && ruff format    # lint + format (also run by pre-commit)
env MYPYPATH=src mypy --explicit-package-bases src   # type check
pre-commit run --all-files         # ruff + mypy + file hygiene
uvicorn snowtool.api.app:get_app --factory --reload   # dev API server (no module-level app; CONTRIBUTING.md is stale)
snowtool --root <db> ...           # the CLI (entrypoint: snowtool.__main__:cli)
```

- Python **3.14+**. Single package: `src/snowtool`. Version is git-tag-derived (hatch-vcs) — don't hand-edit `__version__.py`.
- Runtime config is pydantic-settings with **dotenv disabled**: the env var `SNOWTOOL_SNOWDB_CONFIG` (or `--config`/`-C`) must be set to use the CLI/API. Tests don't need it (they build `Settings(snowdb_config=tmp_path)`).
- No system GDAL: raster I/O is rasterio + async-tiff + griffine, not osgeo bindings.

## Architecture

A **snowdb** is an on-disk directory (`aois/` + `data/<dataset>/` + a root `snowdb_conf.json`) holding multiple gridded snow datasets, queried per basin (AOI). The code splits cleanly into a domain core (`snowdb/`), a thin CLI shell (`cli/`), and a FastAPI read API (`api/`). All three sit on the same `SnowDb`/`Dataset` objects.

**Read vs. write split.** `SnowDb` (`snowdb/db.py`) is the read/query surface; `SnowDbManager` (`snowdb/manager.py`) is the admin/write surface that wraps it (`manager.db`). The CLI's `pass_snowdb`/`pass_manager` decorators pick the right one. Command bodies stay thin — new logic belongs on `SnowDb`/`Dataset` or in `snowdb/diagnostics.py`, never in click callbacks.

**Datasets are data, not subclasses.** A `DatasetSpec` (`snowdb/spec.py`) carries a grid, its `DatasetVariable`s, and an `Ingester`. Built-in specs live in `snowdb/datasets/` (snodas, swann, instarr) and are collected into `DEFAULT_DATASET_SPECS` / `DATASET_TEMPLATES`. Adding a dataset is a new spec + ingester, registered in that package — no new dataset class. Specs reach a `SnowDb` as inline or path config links (`snowdb/config.py`); ingest turns a source artifact into per-date COGs under `data/<name>/cogs/<YYYYMMDD>/`.

**Zone layers are pluggable along three axes** — read these together, they interlock:
- `ZoneLayerProvider` (terrain, landcover) generates derived layers (`snowdb/terrain.py`, `landcover.py`); the built-ins are `DEFAULT_ZONE_LAYER_PROVIDERS` in `zone_layer_providers.py`.
- `ZoneLayerSource` is *where the input comes from* (`DemSource`→`LocalFile`/`ThreeDEP`; `LandCoverSource`→`LocalFile`/`AnnualNLCD`). Sources are **injected** (CLI: `CliContext.zone_layer_sources`), so tests use local files instead of S3.
- `ZoneScheme` (`snowdb/zoning.py`) turns a layer's pixel values into zones: `BandedZoning` (elevation bands), `ThresholdZoning` (forest above/below %), `CategoricalZoning` (aspect classes).

**Query = crossed zonal stats.** `ZonalStats.calculate` (`snowdb/zonal_stats.py`) reduces a `RasterCollection` over an `AOIRaster`, crossing N selected zone axes into a mixed-radix product index (`_ZoneIndex`; K=0 is whole-basin, K=1 single-axis). `Reducer.MEAN` is area-weighted over valid pixels; `Reducer.TOTAL` is the area-weighted basin total. **The AOI raster carries per-pixel cell area inside the basin (0 outside)** — it is both the membership mask and the area weights; elevation/aspect/forest are read live from the zone-layer sets at query time, decoupled from the AOI.

**Provenance is a versioned hash.** Every derived artifact is tagged `v{format_version}:{sha256}` (`snowdb/provenance.py`); a content change *or* a format-version bump reads as stale and forces a rebuild via the same equality check. Tags: `SNOWTOOL_TILE_BBOX`, `SNOWTOOL_AOI_HASH`, `SNOWTOOL_DEM_HASH`, `SNOWTOOL_NLCD_HASH` (`snowdb/constants.py`). This is a greenfield DB — versions start at 1, `migration/` exists only to lift the developer's own pre-rework dev dirs forward.

## Test writing guidelines

The suite favors **real-code integration on a tiny synthetic grid** over isolated units. Match these conventions:

- **Use the synthetic-grid fixtures** in `tests/conftest.py` (`spec`, `grid`, `dataset`, `swe_cog`, `aoi_geojson`, plus `write_terrain`/`write_landcover` helpers). Everything runs on a 512×512 / 2×2-tile geographic grid with hand-computable uniform values, exercising real rasterio/griffine with no system GDAL. CLI tests add `runner`/`cli_obj`/`initialized_root` in `tests/cli/conftest.py` and drive real commands via `runner.invoke(cli, args, obj=cli_obj)`.
- **Prefer one integration test that walks the pipeline** (see `tests/snowdb/test_pipeline.py`) over many unit tests. The bar for a dedicated unit file is high: justify it like `tests/snowdb/test_zonal_stats.py` does — it stubs raster I/O *only* to pin numeric cases (non-uniform pixel area, nodata-in-band, area-weighting) that the uniform pipeline test physically cannot distinguish.
- **Inject, don't monkeypatch.** Zone-layer sources and (preferably) engines reach the code through `CliContext`/constructor seams — pass a `LocalFile`/fake through the seam. `monkeypatch` is acceptable only at true boundaries: env vars for `Settings` (`setenv`/`delenv`), or the network client (`dem_source.TIFF`) when a non-network test needs S3 behavior. Patching a module-global engine is a smell that the seam is missing. Never use `unittest.mock` — the suite has zero mocks and should stay that way.
- **Parametrize repeated cases.** A cluster of tests differing only in inputs/expected (e.g. token parsers, reducer variants, CSV shapes) is one `@pytest.mark.parametrize`, not N copied functions.
- **Assert specific, hand-computable values** (exact `mean_swe_mm`, `area_m2 == aoi_raster.array.sum()`, exact band bounds), not just `exit_code == 0`. For CLI output, prefer asserting `--format json` payloads over substring matches on prose.
- Real-S3 tests go in a `*_network.py` file behind the `@pytest.mark.network` marker (deselected by default).
