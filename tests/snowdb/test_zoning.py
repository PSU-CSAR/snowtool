"""The zone model: BandedZoning / CategoricalZoning assignment + the registry."""

from itertools import pairwise

import numpy
import pytest

from snowtool.snowdb import diagnostics
from snowtool.snowdb.config import ZoneLayerParams
from snowtool.snowdb.constants import M_TO_FT
from snowtool.snowdb.spec import DatasetSpec, GridParams
from snowtool.snowdb.zones.landcover import FOREST_COVER
from snowtool.snowdb.zones.terrain import (
    ASPECT_COMPONENT_NODATA,
    ASPECT_MAJORITY,
    EASTNESS,
    ELEVATION,
    ELEVATION_NODATA,
    NORTHNESS,
)
from snowtool.snowdb.zones.zone_layer import (
    ZoneLayer,
    ZoneLayerProvider,
    ZoneLayerSource,
    available_zones,
)
from snowtool.snowdb.zones.zone_layer_providers import DEFAULT_ZONE_LAYER_PROVIDERS
from snowtool.snowdb.zones.zoning import (
    BandedZoning,
    BandZone,
    CategoricalZoning,
    ClassZone,
    ThresholdZoning,
    ZoneClassDescription,
    ZoneDescription,
)

from ..conftest import make_manager, make_snowdb


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
    bounds = [(z.min, z.max) for z in zones]
    assert bounds == [(0, 1000), (1000, 2000), (2000, 3000)]
    # Contiguous: each band's max is the next band's min.
    assert all(a[1] == b[0] for a, b in pairwise(bounds))


def test_banded_step_override_changes_band_width():
    scheme = _elevation_scheme()
    default_bands = scheme.zones()
    coarser = scheme.with_override(2000).zones()
    assert len(coarser) < len(default_bands)
    assert (coarser[0].min, coarser[0].max) == (-2000, 0)


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


def test_banded_describe_reports_param_default_and_unit():
    scheme = BandedZoning(
        domain_min=0,
        domain_max=2000,
        default_step=1000,
        unit='ft',
        value_scale=M_TO_FT,
        layer_nodata=ELEVATION_NODATA,
    )
    assert scheme.describe() == ZoneDescription(
        kind='banded',
        param_key='band_step_ft',
        default=1000,
        unit='ft',
        classes=None,
    )


# --- aspect-component banding (northness / eastness) -------------------------


def test_aspect_component_bands_span_minus_one_to_one_in_pct():
    scheme = NORTHNESS.zoning
    assert isinstance(scheme, BandedZoning)
    bands = scheme.zones()
    # value_scale 100, default step 50 pct == 0.5 native, aligned to 0: four bands
    # spanning [-1, 1] plus the closed-top boundary band for exactly +1.0.
    assert [(b.min, b.max) for b in bands] == [
        (-100, -50),
        (-50, 0),
        (0, 50),
        (50, 100),
        (100, 150),
    ]
    assert all(b.unit == 'pct' for b in bands)


@pytest.mark.parametrize(
    ('component', 'ordinal'),
    [
        (-0.7, 0),  # -70 pct -> [-100, -50)
        (-0.2, 1),  # -20 pct -> [-50, 0)
        (0.0, 2),  # 0 pct -> [0, 50)
        (0.6, 3),  # 60 pct -> [50, 100)
        (ASPECT_COMPONENT_NODATA, -1),  # finite nodata -> out of zone
    ],
)
def test_aspect_component_assign_bands_native_values(component, ordinal):
    # northness/eastness share the scheme; native cos/sin values in [-1, 1] scale
    # by 100 into percent bands, and the finite nodata sentinel digitises out.
    for scheme in (NORTHNESS.zoning, EASTNESS.zoning):
        assert isinstance(scheme, BandedZoning)
        values = numpy.array([[component]], dtype=numpy.float32)
        numpy.testing.assert_array_equal(scheme.assign(values), [[ordinal]])


def test_aspect_component_band_step_is_overridable():
    scheme = NORTHNESS.zoning
    assert isinstance(scheme, BandedZoning)
    # A 100 pct (== 1.0 native) step collapses each half to a single band.
    coarse = scheme.with_override(100).zones()
    assert [(b.min, b.max) for b in coarse] == [(-100, 0), (0, 100), (100, 200)]
    # The dataset param key that carries this override.
    assert scheme.param_key == 'band_step_pct'


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


def test_categorical_describe_has_no_param_and_lists_classes():
    scheme = ASPECT_MAJORITY.zoning
    assert isinstance(scheme, CategoricalZoning)
    assert scheme.describe() == ZoneDescription(
        kind='categorical',
        param_key=None,
        default=None,
        unit=None,
        classes=(
            ZoneClassDescription(key='N', label='N'),
            ZoneClassDescription(key='E', label='E'),
            ZoneClassDescription(key='S', label='S'),
            ZoneClassDescription(key='W', label='W'),
            ZoneClassDescription(key='flat', label='flat'),
        ),
    )


# --- ThresholdZoning ---------------------------------------------------------


def test_forest_cover_uses_a_threshold_split():
    # Forest cover is a forested/unforested split, not percent bands. Labels are
    # clean; the threshold rides on each zone as a structured value.
    assert isinstance(FOREST_COVER.zoning, ThresholdZoning)
    below, above = FOREST_COVER.zoning.zones()
    assert (below.label, below.side, below.threshold, below.unit) == (
        'unforested',
        'below',
        50,
        '%',
    )
    assert (above.label, above.side, above.threshold, above.unit) == (
        'forested',
        'above',
        50,
        '%',
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
    raised = scheme.with_override(50)
    numpy.testing.assert_array_equal(raised.assign(values), [[0, 1]])
    # The override threshold rides on the zones (labels stay clean).
    below, above = raised.zones()
    assert (below.label, below.threshold) == ('unforested', 50)
    assert (above.label, above.threshold) == ('forested', 50)


def test_threshold_describe_reports_param_default_and_unit():
    scheme = ThresholdZoning(
        default_threshold=40,
        unit='%',
        value_scale=1,
        layer_nodata=255,
        below_label='unforested',
        above_label='forested',
    )
    assert scheme.describe() == ZoneDescription(
        kind='threshold',
        param_key='threshold_pct',
        default=40,
        unit='%',
        classes=None,
    )


# --- the registry ------------------------------------------------------------


def test_available_zones_lists_zoneable_layers_including_components():
    zones = available_zones(DEFAULT_ZONE_LAYER_PROVIDERS)

    assert set(zones) == {
        'terrain.elevation',
        'terrain.aspect',
        'terrain.northness',
        'terrain.eastness',
        'terrain.aspect_entropy',
        'landcover.forest_cover',
    }
    # The aspect-orientation components are now each their own banded axis.
    assert isinstance(zones['terrain.northness'].scheme, BandedZoning)
    assert isinstance(zones['terrain.eastness'].scheme, BandedZoning)
    # Each entry carries the provider, layer, and its scheme.
    elevation = zones['terrain.elevation']
    assert elevation.layer is ELEVATION
    assert elevation.scheme is ELEVATION.zoning


def test_snowdb_available_zones_delegates(tmp_path, spec):

    db = make_snowdb(tmp_path, [spec])
    assert set(db.available_zones()) == {
        'terrain.elevation',
        'terrain.aspect',
        'terrain.northness',
        'terrain.eastness',
        'terrain.aspect_entropy',
        'landcover.forest_cover',
    }


def test_enablement_scopes_providers_generation_and_available_zones(tmp_path):
    # A terrain-only dataset: its zones enable terrain but not land cover.
    terrain_only = DatasetSpec(
        name='terr',
        grid_params=GridParams(
            origin_x=-120.0,
            origin_y=45.0,
            px_size=0.01,
            cols=8,
            rows=8,
            tile_size=8,
        ),
        zones={'terrain': {'elevation': ZoneLayerParams(band_step_ft=1000)}},
    )
    manager = make_manager(tmp_path, [terrain_only])
    db = manager.db
    ds = db['terr']

    # Bound to terrain only -- land cover is neither a provider nor a zone set.
    assert set(ds.zones) == {'terrain'}
    assert 'landcover' not in ds.providers
    # available_zones reflects only what some dataset enables.
    zones = db.available_zones()
    assert 'terrain.elevation' in zones
    assert 'landcover.forest_cover' not in zones
    # Generation for a provider no dataset enables targets nothing.
    assert manager.generate_zone_layers('landcover', db.datasets.values()) == {}


def test_a_new_provider_needs_no_plumbing_edits(tmp_path, spec):
    # A throwaway provider added only to the registry must be visible to every
    # generic seam -- Dataset.zones, artifact_status, diagnostics, the registry --
    # with no edits to Dataset/SnowDb/diagnostics. (Verification #6.)
    class _StubSource(ZoneLayerSource):
        def open(self, bounds):  # pragma: no cover - never opened in this test
            raise NotImplementedError

    class TinyProvider(ZoneLayerProvider):
        name = 'tiny'
        subdir = 'tiny'
        hash_tag = 'SNOWTOOL_TINY_HASH'
        format_version = 1
        layers = (
            ZoneLayer(
                filename='tier.tif',
                dtype='uint8',
                nodata=255,
                band_descriptions=('tier',),
                key='tier',
                zoning=CategoricalZoning(
                    classes=(
                        ClassZone(key='a', label='a', code=0),
                        ClassZone(key='b', label='b', code=1),
                    ),
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
    # The dataset must *enable* the new provider (its zones block) for it to be
    # bound/served -- enablement is opt-in.
    spec.zones = {**spec.zones, 'tiny': {'tier': ZoneLayerParams()}}
    db = make_snowdb(tmp_path, [spec], zone_layer_providers=providers)
    ds = db['test']

    # Bound as a zone-layer set, reported by artifact status + the registry...
    assert 'tiny' in ds.zones
    assert 'tiny' in ds.artifact_status().zone_layers
    assert 'tiny.tier' in db.available_zones()
    # ...and (since it isn't built on disk) surfaced as a missing artifact.
    assert any(m.startswith('tiny') for m in diagnostics.missing_artifacts(ds))
