"""The zone model: BandedZoning / CategoricalZoning assignment + the registry."""

from itertools import pairwise

import numpy

from snowtool.snowdb.constants import M_TO_FT
from snowtool.snowdb.landcover import FOREST_COVER
from snowtool.snowdb.terrain import ASPECT_MAJORITY, ELEVATION, ELEVATION_NODATA
from snowtool.snowdb.zone_layer import available_zones
from snowtool.snowdb.zone_layer_providers import DEFAULT_ZONE_LAYER_PROVIDERS
from snowtool.snowdb.zoning import (
    BandedZoning,
    BandZone,
    CategoricalZoning,
    ThresholdZoning,
)


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


# --- ThresholdZoning ---------------------------------------------------------


def test_forest_cover_uses_a_threshold_split():
    # Forest cover is a forested/unforested split, not percent bands. Labels are
    # clean; the threshold rides on each zone as a structured value.
    assert isinstance(FOREST_COVER.zoning, ThresholdZoning)
    below, above = FOREST_COVER.zoning.zones()
    assert (below.label, below.side, below.threshold, below.unit) == (
        'unforested', 'below', 50, '%',
    )
    assert (above.label, above.side, above.threshold, above.unit) == (
        'forested', 'above', 50, '%',
    )


def test_threshold_assign_splits_below_and_at_or_above():
    scheme = ThresholdZoning(
        default_threshold=40,
        unit='%',
        value_scale=1,
        layer_nodata=255,
        below_label='unforested',
        above_label='forested',
    )
    # 39 -> below (0); 40 -> at-or-above (1); 100 -> above (1); 255 nodata -> -1.
    values = numpy.array([[39, 40, 100, 255]], dtype=numpy.uint8)
    numpy.testing.assert_array_equal(scheme.assign(values), [[0, 1, 1, -1]])


def test_threshold_override_moves_the_split_and_relabels():
    scheme = ThresholdZoning(
        default_threshold=40,
        unit='%',
        value_scale=1,
        layer_nodata=255,
        below_label='unforested',
        above_label='forested',
    )
    values = numpy.array([[40, 60]], dtype=numpy.uint8)
    # With the split raised to 50, the 40% pixel drops below it.
    numpy.testing.assert_array_equal(scheme.assign(values, threshold=50), [[0, 1]])
    # The override threshold rides on the zones (labels stay clean).
    below, above = scheme.zones(threshold=50)
    assert (below.label, below.threshold) == ('unforested', 50)
    assert (above.label, above.threshold) == ('forested', 50)


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


def test_a_new_provider_needs_no_plumbing_edits(tmp_path, spec):
    # A throwaway provider added only to the registry must be visible to every
    # generic seam -- Dataset.zones, artifact_status, diagnostics, the registry --
    # with no edits to Dataset/SnowDb/diagnostics. (Verification #6.)
    from snowtool.snowdb import diagnostics
    from snowtool.snowdb.db import SnowDb
    from snowtool.snowdb.zone_layer import (
        ZoneLayer,
        ZoneLayerProvider,
        ZoneLayerSource,
    )
    from snowtool.snowdb.zone_layer_providers import DEFAULT_ZONE_LAYER_PROVIDERS
    from snowtool.snowdb.zoning import ClassZone, categorical

    class _StubSource(ZoneLayerSource):
        def open(self, bounds):  # pragma: no cover - never opened in this test
            raise NotImplementedError

    class TinyProvider(ZoneLayerProvider):
        name = 'tiny'
        subdir = 'tiny'
        hash_tag = 'SNOWTOOL_TINY_HASH'
        layers = (
            ZoneLayer(
                filename='tier.tif',
                dtype='uint8',
                nodata=255,
                band_descriptions=('tier',),
                key='tier',
                zoning=categorical(
                    (ClassZone(key='a', label='a', code=0),
                     ClassZone(key='b', label='b', code=1)),
                    layer_nodata=255,
                ),
            ),
        )

        def default_source(self, root):
            return _StubSource()

        def local_source(self, path):  # pragma: no cover - not exercised here
            return _StubSource()

        def generate(self, source, targets, bounds, *, force=False, **options):
            return {}

    providers = (*DEFAULT_ZONE_LAYER_PROVIDERS, TinyProvider())
    db = SnowDb(tmp_path, [spec], zone_layer_providers=providers)
    ds = db['test']

    # Bound as a zone-layer set, reported by artifact status + the registry...
    assert 'tiny' in ds.zones
    assert 'tiny' in ds.artifact_status().zone_layers
    assert 'tiny.tier' in db.available_zones()
    # ...and (since it isn't built on disk) surfaced as a missing artifact.
    assert 'tiny' in diagnostics.missing_artifacts(ds)
