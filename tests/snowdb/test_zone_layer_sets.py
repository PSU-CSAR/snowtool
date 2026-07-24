"""The terrain and land-cover ZoneLayerSets: presence, missing layers, hash."""

import pytest

from snowtool.snowdb.raster import TiledRaster
from snowtool.snowdb.zones.landcover import landcover_provider
from snowtool.snowdb.zones.landcover_layers import (
    FOREST_COVER,
    LANDCOVER_FORMAT_VERSION,
)
from snowtool.snowdb.zones.terrain import terrain_provider
from snowtool.snowdb.zones.terrain_layers import (
    ASPECT_MAJORITY,
    TERRAIN_FORMAT_VERSION,
)
from snowtool.snowdb.zones.zone_layer import ZoneLayerSet

_CASES = [
    pytest.param(
        'terrain',
        terrain_provider,
        ASPECT_MAJORITY,
        'aspect_majority.tif',
        TERRAIN_FORMAT_VERSION,
        id='terrain',
    ),
    pytest.param(
        'landcover',
        landcover_provider,
        FOREST_COVER,
        'forest_cover_pct.tif',
        LANDCOVER_FORMAT_VERSION,
        id='landcover',
    ),
]
_AXES = ('zone_key', 'provider_factory', 'layer', 'missing_filename', 'format_version')


@pytest.mark.parametrize(_AXES, _CASES)
def test_present_and_provenance_hash_on_a_built_dataset(
    dataset,
    zone_key,
    provider_factory,
    layer,
    missing_filename,
    format_version,
):
    zone_set = dataset.zones[zone_key]
    assert isinstance(zone_set, ZoneLayerSet)
    assert zone_set.present() is True
    assert zone_set.missing_layers() == []

    digest = zone_set.provenance_hash()
    assert digest is not None
    # Versioned provenance: a vN: prefix in front of the 64-hex sha256.
    version, _, hex_digest = digest.partition(':')
    assert version == f'v{format_version}'
    assert len(hex_digest) == 64


@pytest.mark.parametrize(_AXES, _CASES)
def test_missing_layers_reports_an_absent_layer(
    dataset,
    zone_key,
    provider_factory,
    layer,
    missing_filename,
    format_version,
):
    zone_set = dataset.zones[zone_key]
    zone_set.layer_path(layer).unlink()

    missing = zone_set.missing_layers()

    assert zone_set.present() is False
    assert [layer.filename for layer in missing] == [missing_filename]


@pytest.mark.parametrize(_AXES, _CASES)
def test_hash_is_none_without(
    tmp_path,
    zone_key,
    provider_factory,
    layer,
    missing_filename,
    format_version,
):
    zone_set = provider_factory().layer_set(tmp_path / zone_key)
    assert zone_set.present() is False
    assert zone_set.provenance_hash() is None


@pytest.mark.parametrize(_AXES, _CASES)
def test_raster_points_at(
    dataset,
    zone_key,
    provider_factory,
    layer,
    missing_filename,
    format_version,
):
    zone_set = dataset.zones[zone_key]
    raster = zone_set.raster(layer)
    assert isinstance(raster, TiledRaster)
    assert raster.path == zone_set.directory / layer.filename


def test_tiled_raster_raises_file_not_found_for_a_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match='No such raster'):
        TiledRaster(tmp_path / 'nope.tif')
