"""The land-cover ZoneLayerSet: presence, missing layers, and provenance hash."""

from snowtool.snowdb.landcover import FOREST_COVER, LandCoverProvider
from snowtool.snowdb.raster import TiledRaster
from snowtool.snowdb.zone_layer import ZoneLayerSet


def test_present_and_provenance_hash_on_a_built_dataset(dataset):
    landcover = dataset.zones['landcover']
    assert isinstance(landcover, ZoneLayerSet)
    assert landcover.present() is True
    assert landcover.missing_layers() == []

    digest = landcover.provenance_hash()
    assert digest is not None
    assert len(digest) == 64


def test_missing_layers_reports_an_absent_layer(dataset):
    landcover = dataset.zones['landcover']
    landcover.layer_path(FOREST_COVER).unlink()

    missing = landcover.missing_layers()

    assert landcover.present() is False
    assert [layer.filename for layer in missing] == ['forest_cover_pct.tif']


def test_provenance_hash_is_none_without_landcover(tmp_path):
    landcover = LandCoverProvider().layer_set(tmp_path / 'landcover')
    assert landcover.present() is False
    assert landcover.provenance_hash() is None


def test_raster_points_at_the_layer(dataset):
    landcover = dataset.zones['landcover']
    raster = landcover.raster(FOREST_COVER)
    assert isinstance(raster, TiledRaster)
    assert raster.path == landcover.directory / FOREST_COVER.filename
