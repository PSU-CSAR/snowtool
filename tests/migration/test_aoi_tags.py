import numpy
import pytest
import rasterio

from click.testing import CliRunner

from snowtool.cli import cli
from snowtool.migration.aoi_tags import (
    LEGACY_ORIGIN_TILE_TAG,
    LEGACY_TILE_TAG_PREFIX,
    migrate_aoi_raster_tags,
    quadkey_to_tile_coords,
)
from snowtool.snowdb.aoi_raster import AOIRaster
from snowtool.snowdb.constants import TILE_BBOX_TAG
from snowtool.snowdb.datasets import SNODAS_SPEC
from snowtool.snowdb.grid import tile_base_origin
from snowtool.snowdb.raster.cog import write_cog

SNODAS_GRID = SNODAS_SPEC.grid
TILE = 256


def _write_legacy(path, quadkeys, origin):
    """Write a tiny COG tagged like a legacy snodas AOI raster."""

    write_cog(
        path,
        numpy.zeros((TILE, TILE), dtype=numpy.float32),
        transform=SNODAS_GRID[0, 0].transform,
        tile_size=TILE,
        tags={
            LEGACY_ORIGIN_TILE_TAG: origin,
            **{
                f'{LEGACY_TILE_TAG_PREFIX}_{i:03d}': quadkey
                for i, quadkey in enumerate(quadkeys)
            },
        },
    )


# --- quadkey decoder (the migration's sole remaining use of quadkeys) ---------


@pytest.mark.parametrize(
    ('row', 'col', 'quadkey'),
    [
        (0, 0, '0000'),
        (0, 1, '0001'),
        (1, 0, '0002'),
        (1, 1, '0003'),
        (2, 4, '0120'),
    ],
)
def test_quadkey_decodes_to_tile_coords(row: int, col: int, quadkey: str) -> None:
    assert quadkey_to_tile_coords(quadkey) == (row, col)


def test_quadkey_wrong_zoom_rejected() -> None:
    with pytest.raises(ValueError, match='zoom'):
        quadkey_to_tile_coords('012')  # 3 chars, native zoom is 4


def test_quadkey_invalid_char_rejected() -> None:
    with pytest.raises(ValueError, match='Invalid quadkey'):
        quadkey_to_tile_coords('0009')


# --- migration ----------------------------------------------------------------


def test_migrate_writes_bbox_from_quadkey_set(tmp_path) -> None:
    path = tmp_path / 'legacy.tif'
    # tiles (0,0),(0,1),(1,0),(1,1) -> bbox "0 0 1 1"
    _write_legacy(path, ['0000', '0001', '0002', '0003'], origin='0000')

    assert migrate_aoi_raster_tags(path) is True

    with rasterio.open(path) as ds:
        assert ds.tags()[TILE_BBOX_TAG] == '0 0 1 1'


def test_migrate_is_idempotent(tmp_path) -> None:
    path = tmp_path / 'legacy.tif'
    _write_legacy(path, ['0120'], origin='0120')  # single tile (2, 4)

    assert migrate_aoi_raster_tags(path) is True
    with rasterio.open(path) as ds:
        first = ds.tags()[TILE_BBOX_TAG]

    # Second run is a no-op and leaves the tag unchanged.
    assert migrate_aoi_raster_tags(path) is False
    with rasterio.open(path) as ds:
        assert ds.tags()[TILE_BBOX_TAG] == first == '2 4 2 4'


def test_migrate_missing_tags_raises(tmp_path) -> None:

    path = tmp_path / 'plain.tif'
    write_cog(
        path,
        numpy.zeros((TILE, TILE), dtype=numpy.float32),
        transform=SNODAS_GRID[0, 0].transform,
        tile_size=TILE,
    )

    with pytest.raises(ValueError, match='nothing to migrate'):
        migrate_aoi_raster_tags(path)


def test_migrated_raster_reads_via_bbox(tmp_path) -> None:
    """End-to-end: after migration the runtime reads the same window it did
    from the legacy quadkey tags (the bbox spans the legacy tile set)."""

    path = tmp_path / 'legacy.tif'
    _write_legacy(path, ['0120'], origin='0120')  # tile (2, 4)
    migrate_aoi_raster_tags(path)

    aoi_raster = AOIRaster.open(path, SNODAS_GRID)
    assert [(t.row, t.col) for t in aoi_raster.tiles] == [(2, 4)]
    assert aoi_raster.origin == tile_base_origin(SNODAS_GRID[2, 4])


# --- CLI wiring ---------------------------------------------------------------


def test_cli_migration_aoi_tags(tmp_path) -> None:

    path = tmp_path / 'legacy.tif'
    _write_legacy(path, ['0120'], origin='0120')

    result = CliRunner().invoke(cli, ['migration', 'aoi-tags', str(path)])

    assert result.exit_code == 0, result.output
    assert 'migrated' in result.output
    with rasterio.open(path) as ds:
        assert ds.tags()[TILE_BBOX_TAG] == '2 4 2 4'
