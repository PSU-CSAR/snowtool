"""The terrain ZoneLayerSet: presence, missing layers, and the provenance hash."""

import pytest

from snowtool.snowdb.raster import TiledRaster
from snowtool.snowdb.terrain import (
    ASPECT_MAJORITY,
    ELEVATION,
    TERRAIN_FORMAT_VERSION,
    TerrainProvider,
)
from snowtool.snowdb.zone_layer import ZoneLayerSet


def test_present_and_provenance_hash_on_a_built_dataset(dataset):
    terrain = dataset.zones['terrain']
    assert isinstance(terrain, ZoneLayerSet)
    assert terrain.present() is True
    assert terrain.missing_layers() == []

    digest = terrain.provenance_hash()
    assert digest is not None
    # Versioned provenance: a vN: prefix in front of the 64-hex sha256.
    version, _, hex_digest = digest.partition(':')
    assert version == f'v{TERRAIN_FORMAT_VERSION}'
    assert len(hex_digest) == 64


def test_missing_layers_reports_an_absent_layer(dataset):

    terrain = dataset.zones['terrain']
    terrain.layer_path(ASPECT_MAJORITY).unlink()

    missing = terrain.missing_layers()

    assert terrain.present() is False
    assert [layer.filename for layer in missing] == ['aspect_majority.tif']


def test_provenance_hash_is_none_without_terrain(tmp_path):
    terrain = TerrainProvider().layer_set(tmp_path / 'terrain')
    assert terrain.present() is False
    assert terrain.provenance_hash() is None


def test_raster_points_at_the_elevation_layer(dataset):
    terrain = dataset.zones['terrain']
    raster = terrain.raster(ELEVATION)
    assert isinstance(raster, TiledRaster)
    assert raster.path == terrain.directory / ELEVATION.filename


def test_tiled_raster_raises_file_not_found_for_a_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match='No such raster'):
        TiledRaster(tmp_path / 'nope.tif')
