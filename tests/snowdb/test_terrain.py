"""The TerrainSet reader: presence, missing layers, and the provenance hash."""

import pytest

from snowtool.snowdb.terrain import ELEVATION, ElevationRaster, TerrainSet


def test_present_and_dem_hash_on_a_built_dataset(dataset):
    terrain = dataset.terrain
    assert isinstance(terrain, TerrainSet)
    assert terrain.present() is True
    assert terrain.missing_layers() == []

    digest = terrain.dem_hash()
    assert digest is not None
    assert len(digest) == 64


def test_missing_layers_reports_an_absent_layer(dataset):
    dataset.terrain.aspect_majority_path.unlink()

    missing = dataset.terrain.missing_layers()

    assert dataset.terrain.present() is False
    assert [layer.filename for layer in missing] == ['aspect_majority.tif']


def test_dem_hash_is_none_without_terrain(tmp_path):
    terrain = TerrainSet(tmp_path / 'terrain')
    assert terrain.present() is False
    assert terrain.dem_hash() is None


def test_elevation_raster_points_at_the_elevation_layer(dataset):
    raster = dataset.terrain.elevation_raster()
    assert isinstance(raster, ElevationRaster)
    assert raster.path == dataset.terrain.directory / ELEVATION.filename


def test_tiled_raster_raises_file_not_found_for_a_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match='No such raster file'):
        ElevationRaster(tmp_path / 'nope.tif')
