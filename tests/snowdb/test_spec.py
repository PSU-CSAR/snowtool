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


def test_crs_is_the_single_parsed_grid_crs():
    spec = _projected_spec()
    # One parsed CRS, shared by is_geographic/cell_area and the dataset's
    # rasterio write CRS -- not independently re-parsed from grid_params.
    assert spec.crs is spec.grid.crs
    assert spec.crs.to_epsg() == 32611
    assert spec.crs is spec.crs  # cached


def _named_spec(name: str) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        grid_params=GridParams(
            origin_x=-120.0,
            origin_y=45.0,
            px_size=0.01,
            cols=8,
            rows=8,
            tile_size=8,
        ),
        dem_min_m=0.0,
        dem_max_m=1000.0,
    )


def test_model_prefix_camelcases_and_collapses_separators():
    assert _named_spec('snodas').model_prefix == 'Snodas'
    # Case and -/_ differences collapse to the same prefix (why SnowDb guards
    # against such name collisions).
    assert _named_spec('foo-bar').model_prefix == 'FooBar'
    assert _named_spec('foo_bar').model_prefix == 'FooBar'


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
