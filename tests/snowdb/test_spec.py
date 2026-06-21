"""DatasetSpec: grid construction and the CRS-driven area properties."""

import pytest

from snowtool.snowdb.spec import DatasetSpec, GridParams


def _geographic_spec() -> DatasetSpec:
    return DatasetSpec(
        name='geo',
        grid_params=GridParams(
            origin_x=-120.0,
            origin_y=45.0,
            px_size=0.01,
            cols=512,
            rows=512,
            tile_size=256,
        ),
        dem_min_m=0.0,
        dem_max_m=1000.0,
    )


def _projected_spec() -> DatasetSpec:
    # UTM zone 11N, 1000 m square pixels -> constant 1e6 m^2 cells.
    return DatasetSpec(
        name='utm',
        grid_params=GridParams(
            origin_x=500_000.0,
            origin_y=4_000_000.0,
            px_size=1000.0,
            cols=128,
            rows=128,
            tile_size=64,
            crs=32611,
        ),
        dem_min_m=0.0,
        dem_max_m=1000.0,
    )


def test_grid_is_built_from_params():
    spec = _geographic_spec()
    assert spec.grid.size == (2, 2)
    assert spec.grid.tile_size == (256, 256)
    # cached: same object each access
    assert spec.grid is spec.grid


def test_geographic_spec_has_no_constant_cell_area():
    spec = _geographic_spec()
    assert spec.is_geographic is True
    with pytest.raises(ValueError, match='geographic'):
        _ = spec.cell_area


def test_projected_spec_has_constant_cell_area():
    spec = _projected_spec()
    assert spec.is_geographic is False
    assert spec.cell_area == pytest.approx(1000.0 * 1000.0)
