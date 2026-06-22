"""The LandCoverSet reader: presence, missing layers, and the provenance hash."""

from snowtool.snowdb.landcover import FOREST_COVER, ForestCoverRaster, LandCoverSet


def test_present_and_nlcd_hash_on_a_built_dataset(dataset):
    landcover = dataset.landcover
    assert isinstance(landcover, LandCoverSet)
    assert landcover.present() is True
    assert landcover.missing_layers() == []

    digest = landcover.nlcd_hash()
    assert digest is not None
    assert len(digest) == 64


def test_missing_layers_reports_an_absent_layer(dataset):
    dataset.landcover.forest_cover_path.unlink()

    missing = dataset.landcover.missing_layers()

    assert dataset.landcover.present() is False
    assert [layer.filename for layer in missing] == ['forest_cover_pct.tif']


def test_nlcd_hash_is_none_without_landcover(tmp_path):
    landcover = LandCoverSet(tmp_path / 'landcover')
    assert landcover.present() is False
    assert landcover.nlcd_hash() is None


def test_forest_cover_raster_points_at_the_layer(dataset):
    raster = dataset.landcover.forest_cover_raster()
    assert isinstance(raster, ForestCoverRaster)
    assert raster.path == dataset.landcover.directory / FOREST_COVER.filename
