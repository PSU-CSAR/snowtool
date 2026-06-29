"""Shared fixtures, including the synthetic-grid pipeline fixtures.

Everything runs on a tiny 512x512 (2x2 tile) grid so the full pipeline —
resample, area raster, AOI rasterize, zonal stats — exercises real rasterio /
griffine code on hand-computable data, with no system GDAL and no large inputs.
These live at the top level so both the ``snowdb`` and ``cli`` suites reuse them.
"""

import hashlib
import json

import numpy
import pytest
import rasterio

from rasterio.crs import CRS

from snowtool.settings import Settings
from snowtool.snowdb.cog import write_cog
from snowtool.snowdb.constants import DEM_HASH_TAG, NLCD_HASH_TAG
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.datasets import SNODAS_VARIABLES
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.landcover import FOREST_COVER, LANDCOVER_FORMAT_VERSION
from snowtool.snowdb.provenance import versioned_hash
from snowtool.snowdb.spec import DatasetSpec, GridParams
from snowtool.snowdb.terrain import (
    ASPECT_COMPONENTS,
    ASPECT_FLAT,
    ASPECT_MAJORITY,
    ELEVATION,
    TERRAIN_FORMAT_VERSION,
)

# gazebo's pytest plugin (assert_problem / assert_has_link / drive_pagination +
# fixtures) is opt-in, not auto-registered.
pytest_plugins = ['gazebo.testing']

# Small synthetic grid parameters.
ORIGIN_X = -120.0
ORIGIN_Y = 45.0
PX = 0.01
SIZE = 512
TILE = 256

DEM_ELEVATION_M = 1000.0  # uniform; 1000 m -> ~3280 ft -> band (3000, 4000) ft
DEM_NODATA = -9999.0
SWE_VALUE = 50  # uniform int16 SWE value
NLCD_FOREST_CLASS = 42  # evergreen forest (in FOREST_CLASSES)
NLCD_NONFOREST_CLASS = 81  # pasture/hay (not forest)
FOREST_PCT_VALUE = 100  # uniform all-forest synthetic land cover


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    return Settings(snowdb_config=tmp_path)


def make_snowdb(root, specs, **kwargs):
    """Build a SnowDb in code from inline dataset configs (no files written).

    The config-object replacement for the old spec-injection constructor: it
    builds a RootConfig rooted at ``root`` (its ``path`` set but the file not
    written) with each spec embedded as an inline dataset, then constructs the
    SnowDb directly. Use this where a test wants a bound SnowDb without staging
    config files on disk.
    """
    from pathlib import Path

    from snowtool.snowdb.config import CONFIG_FILENAME, InlineDatasetLink, RootConfig
    from snowtool.snowdb.datasets import config_from_spec

    config = RootConfig.create()
    config.path = Path(root) / CONFIG_FILENAME
    for spec in specs:
        config.datasets[spec.name] = InlineDatasetLink(dataset=config_from_spec(spec))
    return SnowDb(config, **kwargs)


def make_manager(root, specs, **kwargs):
    """Build a SnowDbManager over an in-code SnowDb (see :func:`make_snowdb`).

    The write-side counterpart to ``make_snowdb``: where a test exercises a
    management op, wrap the inline-config SnowDb in a manager so writes go through
    the admin surface while reads stay on ``manager.db``.
    """
    from snowtool.snowdb.manager import SnowDbManager

    return SnowDbManager(make_snowdb(root, specs, **kwargs))


def register_dataset_config(manager, name, config):
    """Stage ``config`` at ``data/<name>/dataset.json`` and register its link."""
    from snowtool.snowdb.config import DATASET_CONFIG_FILENAME

    ds_dir = manager.db.dataset_dir(name, config)
    ds_dir.mkdir(parents=True, exist_ok=True)
    config_path = ds_dir / DATASET_CONFIG_FILENAME
    config.save(config_path)
    manager.register_dataset(name, config_path)
    return config_path


def init_with_builtins(root):
    """Initialize ``root`` with every built-in dataset registered (from templates)."""
    from snowtool.snowdb.datasets import DATASET_TEMPLATES
    from snowtool.snowdb.manager import SnowDbManager

    manager = SnowDbManager.initialize(root)
    for name, config in DATASET_TEMPLATES.items():
        register_dataset_config(manager, name, config)
    return manager


def populate_synthetic_root(
    root,
    spec,
    pourpoint_geojson,
    *,
    rasterize=True,
    ingest=True,
):
    """Populate ``root`` end-to-end with the synthetic ``spec`` dataset.

    The shared builder behind the reader and API stats tests: it initializes the
    root, registers + binds the dataset, imports the AOI, writes uniform terrain +
    land cover, burns the AOI raster, and (optionally) ingests a uniform SWE COG
    for 2018-04-27. Returns the catalog ``SnowDb`` (its ``root`` is ``root``).
    """
    from datetime import date

    import numpy

    from snowtool.snowdb.cog import write_cog
    from snowtool.snowdb.datasets import config_from_spec
    from snowtool.snowdb.manager import SnowDbManager
    from snowtool.snowdb.pourpoint import Pourpoint

    manager = SnowDbManager.initialize(root)
    register_dataset_config(manager, spec.name, config_from_spec(spec))
    # Reopen so the freshly-registered dataset is bound.
    manager = SnowDbManager.open(root)
    manager.import_pourpoints(pourpoint_geojson)
    dataset = manager.db[spec.name]
    write_terrain(dataset)
    write_landcover(dataset)
    if rasterize:
        dataset.rasterize_aoi(Pourpoint.from_geojson(pourpoint_geojson), force=True)
    if ingest:
        out_dir = dataset.date_dir(date(2018, 4, 27))
        out_dir.mkdir(parents=True, exist_ok=True)
        write_cog(
            out_dir / f'{snodas_swe_name()}.tif',
            numpy.full((SIZE, SIZE), SWE_VALUE, dtype=numpy.int16),
            transform=dataset.grid.base_grid.transform,
            tile_size=TILE,
            predictor=2,
        )
    return manager.db


@pytest.fixture
def spec():
    """A tiny synthetic DatasetSpec (2x2 tile geographic grid)."""
    return DatasetSpec(
        name='test',
        grid_params=GridParams(
            origin_x=ORIGIN_X,
            origin_y=ORIGIN_Y,
            px_size=PX,
            cols=SIZE,
            rows=SIZE,
            tile_size=TILE,
        ),
        variables=SNODAS_VARIABLES,
    )


@pytest.fixture
def grid(spec):
    return spec.grid


@pytest.fixture
def source_dem(tmp_path, grid):
    """A uniform-elevation source DEM on the grid extent."""
    path = tmp_path / 'source_dem.tif'
    array = numpy.full((SIZE, SIZE), DEM_ELEVATION_M, dtype=numpy.float32)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=SIZE,
        width=SIZE,
        count=1,
        dtype='float32',
        crs=CRS.from_epsg(4326),
        transform=grid.base_grid.transform,
        nodata=DEM_NODATA,
    ) as dst:
        dst.write(array, 1)
    return path


def write_uniform_terrain(
    directory,
    *,
    base_grid,
    crs,
    tile_size: int,
    elevation_value: float = DEM_ELEVATION_M,
) -> str:
    """Write a uniform terrain set (elevation + flat aspect) into ``directory``.

    The shared primitive behind both the ``dataset`` fixture's :func:`write_terrain`
    and the CLI suite's fake terrain engine: given the raw grid pieces, it writes
    what :func:`generate_terrain` produces -- uniform elevation, all-flat aspect --
    and stamps the DEM provenance hash on every layer. Returns that hash.
    """
    directory.mkdir(parents=True, exist_ok=True)
    shape = (base_grid.rows, base_grid.cols)

    elevation = numpy.full(shape, elevation_value, dtype='float32')
    dem_hash = versioned_hash(
        TERRAIN_FORMAT_VERSION,
        hashlib.sha256(elevation.tobytes()).hexdigest(),
    )
    common = {
        'transform': base_grid.transform,
        'crs': crs,
        'tile_size': tile_size,
        'tags': {DEM_HASH_TAG: dem_hash},
    }

    write_cog(
        directory / ELEVATION.filename,
        elevation,
        nodata=ELEVATION.nodata,
        band_descriptions=ELEVATION.band_descriptions,
        **common,
    )
    write_cog(
        directory / ASPECT_MAJORITY.filename,
        numpy.full(shape, ASPECT_FLAT, dtype='uint8'),
        nodata=ASPECT_MAJORITY.nodata,
        band_descriptions=ASPECT_MAJORITY.band_descriptions,
        **common,
    )
    write_cog(
        directory / ASPECT_COMPONENTS.filename,
        numpy.full((2, *shape), numpy.nan, dtype='float32'),
        nodata=ASPECT_COMPONENTS.nodata,
        compute_stats=False,
        band_descriptions=ASPECT_COMPONENTS.band_descriptions,
        **common,
    )
    return dem_hash


def write_uniform_landcover(
    directory,
    *,
    base_grid,
    crs,
    tile_size: int,
    pct: int = FOREST_PCT_VALUE,
) -> str:
    """Write a uniform percent-forest land-cover layer into ``directory``.

    The shared primitive behind the ``dataset`` fixture's :func:`write_landcover`
    and the CLI suite's fake land-cover engine. Returns the NLCD provenance hash.
    """
    directory.mkdir(parents=True, exist_ok=True)
    shape = (base_grid.rows, base_grid.cols)

    forest = numpy.full(shape, pct, dtype='uint8')
    nlcd_hash = versioned_hash(
        LANDCOVER_FORMAT_VERSION,
        hashlib.sha256(forest.tobytes()).hexdigest(),
    )

    write_cog(
        directory / FOREST_COVER.filename,
        forest,
        transform=base_grid.transform,
        crs=crs,
        tile_size=tile_size,
        nodata=FOREST_COVER.nodata,
        tags={NLCD_HASH_TAG: nlcd_hash},
        band_descriptions=FOREST_COVER.band_descriptions,
    )
    return nlcd_hash


def write_terrain(dataset, elevation_value: float = DEM_ELEVATION_M) -> str:
    """Write a uniform terrain set onto a dataset's grid (no engine run).

    Convenience wrapper over :func:`write_uniform_terrain` for tests that hold a
    ``dataset`` and just need terrain present (e.g. elevation banding).
    """
    return write_uniform_terrain(
        dataset.zones['terrain'].directory,
        base_grid=dataset.grid.base_grid,
        crs=dataset.grid_crs,
        tile_size=dataset.spec.grid_params.tile_size,
        elevation_value=elevation_value,
    )


def write_landcover(dataset, pct: int = FOREST_PCT_VALUE) -> str:
    """Write a uniform land-cover layer onto a dataset's grid (no engine run)."""
    return write_uniform_landcover(
        dataset.zones['landcover'].directory,
        base_grid=dataset.grid.base_grid,
        crs=dataset.grid_crs,
        tile_size=dataset.spec.grid_params.tile_size,
        pct=pct,
    )


@pytest.fixture
def source_nlcd(tmp_path, grid):
    """A synthetic NLCD land-cover source on the grid extent.

    The left half is forest (class 42), the right half non-forest (class 81), so a
    cell-fraction reduction has a hand-computable, non-uniform result.
    """
    path = tmp_path / 'source_nlcd.tif'
    array = numpy.full((SIZE, SIZE), NLCD_NONFOREST_CLASS, dtype=numpy.uint8)
    array[:, : SIZE // 2] = NLCD_FOREST_CLASS
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=SIZE,
        width=SIZE,
        count=1,
        dtype='uint8',
        crs=CRS.from_epsg(4326),
        transform=grid.base_grid.transform,
        nodata=0,
    ) as dst:
        dst.write(array, 1)
    return path


@pytest.fixture
def dataset(tmp_path, spec):
    """A fully created Dataset: directory, area raster, terrain + land-cover sets."""
    ds = Dataset.create(spec, tmp_path / 'db')
    write_terrain(ds)
    write_landcover(ds)
    return ds


@pytest.fixture
def pourpoint_geojson(tmp_path):
    """A pourpoint with a polygon inside tile (0, 0)."""
    # lon -119.9..-119.0, lat 44.9..44.0 -> well inside the first tile.
    polygon = {
        'type': 'Polygon',
        'coordinates': [
            [
                [-119.9, 44.9],
                [-119.0, 44.9],
                [-119.0, 44.0],
                [-119.9, 44.0],
                [-119.9, 44.9],
            ],
        ],
    }
    point = {'type': 'Point', 'coordinates': [-119.45, 44.45]}
    feature = {
        'type': 'GeometryCollection',
        'id': '12345:MT:USGS',
        'geometries': [point, polygon],
        'properties': {'name': 'Test Basin', 'source': 'test'},
    }
    path = tmp_path / 'pourpoint.geojson'
    path.write_text(json.dumps(feature))
    return path


def snodas_swe_name(date_str: str = '20180427') -> str:
    """A filename matching the SNODAS SWE regex + product glob."""
    # region=us model=ssm datatype=v1 code=1034 scaled=S vcode=lL00
    # T timecode=0001 TTNATS <date> hour=05 interval=H offset=P001
    return f'us_ssmv11034SlL00T0001TTNATS{date_str}05HP001'


@pytest.fixture
def swe_cog(dataset, grid):
    """Write a uniform SWE COG for 2018-04-27 into the db's cogs dir."""
    date_str = '20180427'
    out_dir = dataset._cogs / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{snodas_swe_name(date_str)}.tif'
    array = numpy.full((SIZE, SIZE), SWE_VALUE, dtype=numpy.int16)
    write_cog(
        path,
        array,
        transform=grid.base_grid.transform,
        tile_size=TILE,
        predictor=2,
    )
    return path
