"""The zone model: BandedZoning / CategoricalZoning assignment + the registry."""

from itertools import pairwise

import numpy

from snowtool.snowdb.constants import M_TO_FT
from snowtool.snowdb.terrain import ASPECT_MAJORITY, ELEVATION, ELEVATION_NODATA
from snowtool.snowdb.zone_layer import available_zones
from snowtool.snowdb.zone_layer_providers import DEFAULT_ZONE_LAYER_PROVIDERS
from snowtool.snowdb.zoning import BandedZoning, BandZone, CategoricalZoning


def _elevation_scheme() -> BandedZoning:
    scheme = ELEVATION.zoning
    assert isinstance(scheme, BandedZoning)
    return scheme


# --- BandedZoning ------------------------------------------------------------


def test_banded_zones_are_contiguous_and_aligned_to_zero():
    scheme = BandedZoning(
        domain_min=0,
        domain_max=2000,
        default_step=1000,
        unit='ft',
        value_scale=M_TO_FT,
        layer_nodata=ELEVATION_NODATA,
    )
    zones = scheme.zones()

    assert all(isinstance(z, BandZone) for z in zones)
    bounds = [(z.min, z.max) for z in zones]  # type: ignore[attr-defined]
    assert bounds == [(0, 1000), (1000, 2000), (2000, 3000)]
    # Contiguous: each band's max is the next band's min.
    assert all(a[1] == b[0] for a, b in pairwise(bounds))


def test_banded_step_override_changes_band_width():
    scheme = _elevation_scheme()
    default_bands = scheme.zones()
    coarser = scheme.zones(step=2000)
    assert len(coarser) < len(default_bands)
    assert (coarser[0].min, coarser[0].max) == (-2000, 0)  # type: ignore[attr-defined]


def test_banded_assign_scales_meters_to_feet():
    scheme = BandedZoning(
        domain_min=0,
        domain_max=2000,
        default_step=1000,
        unit='ft',
        value_scale=M_TO_FT,
        layer_nodata=ELEVATION_NODATA,
    )
    # 100 m ~ 328 ft -> band 0; 400 m ~ 1312 ft -> band 1; 700 m ~ 2297 ft -> band 2.
    values = numpy.array([[100.0, 400.0, 700.0]], dtype=numpy.float32)
    ordinals = scheme.assign(values)
    numpy.testing.assert_array_equal(ordinals, [[0, 1, 2]])


def test_banded_assign_marks_nodata_and_out_of_domain_as_minus_one():
    scheme = BandedZoning(
        domain_min=0,
        domain_max=1000,
        default_step=1000,
        unit='ft',
        value_scale=M_TO_FT,
        layer_nodata=ELEVATION_NODATA,
    )
    # nodata sentinel, and 5000 m (~16404 ft) far above the 0..2000 ft domain.
    values = numpy.array([[ELEVATION_NODATA, 5000.0]], dtype=numpy.float32)
    ordinals = scheme.assign(values)
    numpy.testing.assert_array_equal(ordinals, [[-1, -1]])


# --- CategoricalZoning -------------------------------------------------------


def test_categorical_assign_maps_codes_to_ordinals():
    scheme = ASPECT_MAJORITY.zoning
    assert isinstance(scheme, CategoricalZoning)
    # codes 0 N, 1 E, 2 S, 3 W, 4 flat -> ordinals 0..4; 255 nodata -> -1.
    values = numpy.array([[0, 1, 2, 3, 4, 255]], dtype=numpy.uint8)
    ordinals = scheme.assign(values)
    numpy.testing.assert_array_equal(ordinals, [[0, 1, 2, 3, 4, -1]])


def test_categorical_zones_are_the_class_list_in_order():
    scheme = ASPECT_MAJORITY.zoning
    assert isinstance(scheme, CategoricalZoning)
    labels = [z.label for z in scheme.zones()]
    assert labels == ['N', 'E', 'S', 'W', 'flat']


# --- the registry ------------------------------------------------------------


def test_available_zones_lists_zoneable_layers_and_excludes_components():
    zones = available_zones(DEFAULT_ZONE_LAYER_PROVIDERS)

    assert set(zones) == {
        'terrain.elevation',
        'terrain.aspect',
        'landcover.forest_cover',
    }
    # aspect_components has zoning=None, so it never appears.
    assert 'terrain.aspect_components' not in zones
    # Each entry carries the provider, layer, and its scheme.
    elevation = zones['terrain.elevation']
    assert elevation.layer is ELEVATION
    assert elevation.scheme is ELEVATION.zoning


def test_snowdb_available_zones_delegates(tmp_path, spec):
    from snowtool.snowdb.db import SnowDb

    db = SnowDb(tmp_path, [spec])
    assert set(db.available_zones()) == {
        'terrain.elevation',
        'terrain.aspect',
        'landcover.forest_cover',
    }
