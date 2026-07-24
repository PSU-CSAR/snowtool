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

from snowtool.exceptions import NodataMaskError
from snowtool.snowdb.aoi_raster import aoi_provenance
from snowtool.snowdb.config import (
    DATASET_CONFIG_FILENAME,
    DatasetConfig,
)
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.datasets import config_from_spec, template_nodata_mask
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.grid import GridParams
from snowtool.snowdb.manager import SnowDbManager
from snowtool.snowdb.pourpoint import Pourpoint

from ..conftest import ORIGIN_X, ORIGIN_Y, PX, SIZE, TILE, make_dataset

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
    ds = make_dataset(spec, tmp_path / 'db')
    assert ds.nodata_mask is None


def test_aoi_provenance_unchanged_without_mask():
    # Maskless tags must not change: existing AOI rasters stay current and
    # AOI_RASTER_FORMAT_VERSION stays 1.
    assert aoi_provenance('abc', None) == 'v1:abc'
    assert aoi_provenance('abc', 'def') == 'v1:abc+def'


def test_mask_add_change_remove_marks_aoi_stale(
    tmp_path,
    spec,
    grid,
    pourpoint_geojson,
    nodata_mask,
):
    pp = Pourpoint.from_geojson(pourpoint_geojson)
    root = tmp_path / 'db'

    unmasked = make_dataset(spec, root)
    unmasked.rasterize_aoi(pp)
    assert unmasked.aoi_raster_is_current(pp)

    # Adding a mask: the same on-disk raster reads as stale.
    masked = Dataset(spec, root, nodata_mask=nodata_mask)
    assert not masked.aoi_raster_is_current(pp)
    assert masked.rasterize_aoi(pp)
    assert masked.aoi_raster_is_current(pp)

    # The rebuilt raster is stale again from a maskless dataset's view...
    assert not Dataset(spec, root).aoi_raster_is_current(pp)

    # ...and from a dataset whose mask bytes changed. (Fresh instance: the
    # hash is cached per Dataset instance.)
    write_mask(nodata_mask, grid, valid_through_col=99)
    changed = Dataset(spec, root, nodata_mask=nodata_mask)
    assert not changed.aoi_raster_is_current(pp)
    assert changed.rasterize_aoi(pp)
    assert changed.aoi_raster_is_current(pp)


def test_mask_burns_zero_area_outside_domain(
    tmp_path,
    spec,
    pourpoint_geojson,
    nodata_mask,
):
    pp = Pourpoint.from_geojson(pourpoint_geojson)
    unmasked_ds = make_dataset(spec, tmp_path / 'plain')
    unmasked_ds.rasterize_aoi(pp)
    unmasked = unmasked_ds.load_aoi_raster(pp.station_triplet)
    masked_ds = make_dataset(
        spec,
        tmp_path / 'masked',
        nodata_mask=nodata_mask,
    )
    masked_ds.rasterize_aoi(pp)
    masked = masked_ds.load_aoi_raster(pp.station_triplet)

    # Same polygon, same grid -> same tile window; the polygon sits in tile
    # (0, 0), so window col == grid col. The mask zeroes every grid col >=
    # MASK_BOUNDARY_COL; the masked burn must equal the unmasked burn with
    # those columns zeroed, and nothing else may differ.
    expected = unmasked.array.copy()
    expected[:, MASK_BOUNDARY_COL:] = 0
    numpy.testing.assert_array_equal(masked.array, expected)

    # The cut is real on both sides: area strictly between 0 and the full burn.
    assert 0 < masked.array.sum() < unmasked.array.sum()


def test_missing_mask_file_raises_nodata_mask_error(tmp_path, spec, pourpoint_geojson):
    pp = Pourpoint.from_geojson(pourpoint_geojson)
    missing = tmp_path / 'missing-mask.tif'
    ds = make_dataset(spec, tmp_path / 'db', nodata_mask=missing)
    with pytest.raises(NodataMaskError, match='nodata_mask'):
        ds.rasterize_aoi(pp)


def test_mask_shape_mismatch_raises(tmp_path, spec, grid, pourpoint_geojson):
    pp = Pourpoint.from_geojson(pourpoint_geojson)
    bad = tmp_path / 'bad-mask.tif'
    array = numpy.ones((SIZE // 2, SIZE // 2), dtype=numpy.uint8)
    with rasterio.open(
        bad,
        'w',
        driver='GTiff',
        height=SIZE // 2,
        width=SIZE // 2,
        count=1,
        dtype='uint8',
        crs=CRS.from_epsg(4326),
        transform=grid.base_grid.transform,
        nodata=0,
    ) as dst:
        dst.write(array, 1)

    ds = make_dataset(spec, tmp_path / 'db', nodata_mask=bad)
    with pytest.raises(NodataMaskError, match='does not match the dataset grid'):
        ds.rasterize_aoi(pp)


@pytest.mark.parametrize(
    ('template', 'shape'),
    [
        ('snodas', (3351, 6935)),
        ('swann-800m', (3105, 7025)),
    ],
)
def test_template_mask_is_packaged(template, shape):
    packaged = template_nodata_mask(template)
    assert packaged is not None
    assert packaged.is_file()
    with rasterio.open(packaged) as ds:
        assert ds.shape == shape
        assert ds.nodata == 0
    # A template with no packaged mask returns None.
    assert template_nodata_mask('instarr') is None


def test_create_dataset_stamps_mask(tmp_path, spec, nodata_mask):
    # create_dataset with a mask source: the file is copied beside the config,
    # the written config references it, and a fresh SnowDb.open resolves it
    # onto the Dataset (the full round trip a template stamp relies on).
    root = tmp_path / 'db'
    manager = SnowDbManager.initialize(root)
    config = config_from_spec(spec)
    manager.create_dataset('test', config, nodata_mask_source=nodata_mask)

    dataset_dir = root / 'data' / 'test'
    assert (dataset_dir / 'nodata-mask.tif').is_file()

    saved = DatasetConfig.load(dataset_dir / DATASET_CONFIG_FILENAME)
    assert saved.nodata_mask == Path('nodata-mask.tif')

    db = SnowDb.open(root)
    assert db.registered['test'].nodata_mask == dataset_dir / 'nodata-mask.tif'
