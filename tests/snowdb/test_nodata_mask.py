"""Dataset nodata-mask: config resolution, AOI provenance, and the burn.

The mask is a single-band raster on the dataset's full grid whose 0/nodata
pixels are outside the dataset's valid domain (e.g. SNODAS open water). It is
declared per dataset in the config (``nodata_mask``), folded into AOI
provenance so adding/changing/removing it marks AOI rasters stale, and burned
into the AOI raster as zero area weight.
"""

from pathlib import Path

import numpy
import pytest
import rasterio

from rasterio.crs import CRS

from snowtool.snowdb.config import (
    DATASET_CONFIG_FILENAME,
    DatasetConfig,
)
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.grid import GridParams
from snowtool.snowdb.manager import SnowDbManager

from ..conftest import ORIGIN_X, ORIGIN_Y, PX, SIZE, TILE

# The synthetic pourpoint polygon spans lon -119.9..-119.0 (grid cols 10..100 at
# PX=0.01 from ORIGIN_X=-120). Masking everything east of lon -119.45 (col 55)
# cuts the basin mid-polygon, so the burn test has in- and out-of-domain pixels.
MASK_BOUNDARY_COL = 55


def write_mask(path, grid, *, valid_through_col: int = MASK_BOUNDARY_COL):
    """A 0/1 uint8 mask on the full grid: cols < ``valid_through_col`` valid."""
    array = numpy.zeros((SIZE, SIZE), dtype=numpy.uint8)
    array[:, :valid_through_col] = 1
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
def nodata_mask(tmp_path, grid):
    return write_mask(tmp_path / 'nodata-mask.tif', grid)


def _grid_params():
    return GridParams(
        origin_x=ORIGIN_X,
        origin_y=ORIGIN_Y,
        px_size=PX,
        cols=SIZE,
        rows=SIZE,
        tile_size=TILE,
    )


def test_dataset_config_round_trips_nodata_mask(tmp_path):
    config = DatasetConfig(
        grid=_grid_params(),
        variables={},
        nodata_mask=Path('nodata-mask.tif'),
    )
    path = tmp_path / 'dataset.json'
    config.save(path)
    assert DatasetConfig.load(path).nodata_mask == Path('nodata-mask.tif')


def test_dataset_config_nodata_mask_defaults_none():
    config = DatasetConfig(grid=_grid_params(), variables={})
    assert config.nodata_mask is None


def test_snowdb_resolves_nodata_mask_against_dataset_config_dir(
    tmp_path,
    spec,
    grid,
):
    # A referenced (path-link) dataset whose config declares a *relative*
    # nodata_mask: it must resolve against the dataset config's own directory,
    # the same base data_dir resolves against.
    root = tmp_path / 'db'
    manager = SnowDbManager.initialize(root)
    dataset_dir = root / 'data' / 'test'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    mask_path = write_mask(dataset_dir / 'nodata-mask.tif', grid)

    config = DatasetConfig(
        grid=spec.grid_params,
        variables={},
        nodata_mask=Path('nodata-mask.tif'),
    )
    config_path = dataset_dir / DATASET_CONFIG_FILENAME
    config.save(config_path)
    manager.register_dataset('test', config_path)

    db = SnowDb.open(root)
    assert db.registered['test'].nodata_mask == mask_path


def test_dataset_without_mask_has_none(tmp_path, spec):
    ds = Dataset.create(spec, tmp_path / 'db')
    assert ds.nodata_mask is None
