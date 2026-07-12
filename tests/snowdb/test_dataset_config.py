"""The typed dataset config: byte-equal round-trips, the union, ingester resolve."""

import pytest

from pydantic import ValidationError

from snowtool.snowdb.config import (
    BandStepParams,
    BucketParams,
    DatasetConfig,
    EntropyThresholdParams,
    RootConfig,
    ThresholdParams,
    load_entity,
)
from snowtool.snowdb.datasets import (
    DATASET_TEMPLATES,
    DEFAULT_DATASET_SPECS,
    SwannIngester,
    config_from_spec,
)
from snowtool.snowdb.grid import GridParams
from snowtool.snowdb.spec import DEFAULT_ZONES, DatasetSpec
from snowtool.snowdb.variables import DatasetVariable, Reducer, Unit


def _grid_config() -> GridParams:
    return GridParams(
        origin_x=-120.0,
        origin_y=45.0,
        px_size=0.01,
        cols=256,
        rows=256,
        tile_size=256,
    )


def _assert_spec_equivalent(resolved: DatasetSpec, spec: DatasetSpec) -> None:
    """Field-by-field equivalence (DatasetSpec has no ``__eq__``)."""
    assert resolved.name == spec.name
    assert resolved.grid_params == spec.grid_params
    assert resolved.zones == spec.zones
    assert type(resolved.ingester) is type(spec.ingester)
    assert list(resolved.variables) == list(spec.variables)
    for key, variable in spec.variables.items():
        assert resolved.variables[key] == variable
    if spec.footprint is None:
        assert resolved.footprint is None
    else:
        # Exact model equality (footprints are geojson-pydantic geometries now, so
        # the round-trip must not perturb the served footprint at all).
        assert resolved.footprint is not None
        assert resolved.footprint == spec.footprint


@pytest.mark.parametrize('spec', DEFAULT_DATASET_SPECS, ids=lambda s: s.name)
def test_builtin_template_resolves_byte_equal(spec):
    # The template is keyed by dataset name and resolves back to today's spec.
    config = DATASET_TEMPLATES[spec.name]
    _assert_spec_equivalent(DatasetSpec.from_config(config, spec.name), spec)


@pytest.mark.parametrize('spec', DEFAULT_DATASET_SPECS, ids=lambda s: s.name)
def test_builtin_config_round_trips_through_json(tmp_path, spec):
    # Through disk + the discriminated union: save -> load_entity -> from_config.
    config = config_from_spec(spec)
    path = tmp_path / 'dataset.json'
    config.save(path)

    loaded = load_entity(path)

    assert isinstance(loaded, DatasetConfig)
    _assert_spec_equivalent(DatasetSpec.from_config(loaded, spec.name), spec)


def test_ingester_name_resolves_to_the_right_code():

    config = config_from_spec(DEFAULT_DATASET_SPECS[0])  # snodas
    swann_config = DATASET_TEMPLATES['swann-800m']

    assert swann_config.ingester == 'swann'
    resolved = DatasetSpec.from_config(swann_config, 'swann-800m')
    assert isinstance(resolved.ingester, SwannIngester)
    # And snodas names its own ingester kind.
    assert config.ingester == 'snodas'


def test_unknown_ingester_name_is_a_clean_error():
    config = DatasetConfig(
        grid=DATASET_TEMPLATES['snodas'].grid,
        variables=DATASET_TEMPLATES['snodas'].variables,
        ingester='nope',
    )
    with pytest.raises(ValueError, match='unknown ingester'):
        DatasetSpec.from_config(config, 'snodas')


def test_read_only_dataset_has_no_ingester():
    config = DatasetConfig(
        grid=_grid_config(),
        variables={
            'swe': DatasetVariable(
                key='swe',
                unit=Unit(name='mm', scale_factor=1),
                reducer=Reducer.MEAN,
                dtype='int16',
                nodata=-999,
                glob='swe.tif',
            ),
        },
        ingester=None,
    )
    spec = DatasetSpec.from_config(config, 'derived')
    assert spec.ingester is None
    assert spec.footprint is None  # omitted -> serves the whole grid


def test_union_discriminates_root_vs_dataset(tmp_path):
    root = RootConfig.create()
    root_path = tmp_path / 'snowdb_conf.json'
    root.save(root_path)

    assert isinstance(load_entity(root_path), RootConfig)
    ds_path = tmp_path / 'dataset.json'
    config_from_spec(DEFAULT_DATASET_SPECS[0]).save(ds_path)
    assert isinstance(load_entity(ds_path), DatasetConfig)


def test_union_rejects_an_unknown_resource(tmp_path):
    path = tmp_path / 'x.json'
    path.write_text('{"resource": "snowtool.unknown/v1"}')
    with pytest.raises(ValidationError):
        load_entity(path)


def _swe_variable() -> DatasetVariable:
    return DatasetVariable(
        key='swe',
        unit=Unit(name='mm', scale_factor=1),
        reducer=Reducer.MEAN,
        dtype='int16',
        nodata=-999,
        glob='swe.tif',
    )


def test_dataset_variable_rejects_a_dtype_numpy_cannot_parse():
    # A typo'd dtype ('flot32') otherwise validates and only fails at first
    # raster read; parse it at config load instead.
    with pytest.raises(ValidationError, match='dtype'):
        DatasetVariable(
            key='swe',
            unit=Unit(name='mm', scale_factor=1),
            reducer=Reducer.MEAN,
            dtype='flot32',
            nodata=-9999,
            glob='*__swe.tif',
        )


def test_variable_key_is_injected_from_map_key_and_omitted_from_json():
    # The on-disk value carries no 'key'; it comes from the map key on load.
    config = DatasetConfig(grid=_grid_config(), variables={'swe': _swe_variable()})
    dumped = config.model_dump()
    assert 'key' not in dumped['variables']['swe']

    reloaded = DatasetConfig.model_validate_json(config.model_dump_json())
    assert reloaded.variables['swe'].key == 'swe'
    assert reloaded.variables['swe'] == _swe_variable()


def test_variable_instance_whose_key_disagrees_with_map_key_is_rejected():
    with pytest.raises(ValidationError, match='does not match'):
        DatasetConfig(grid=_grid_config(), variables={'depth': _swe_variable()})


def test_unknown_zone_param_is_rejected_at_config_load():
    # extra='forbid' turns a typo'd/unknown zone param into a load-time error.
    with pytest.raises(ValidationError):
        DatasetConfig(
            grid=_grid_config(),
            variables={'swe': _swe_variable()},
            zones={'terrain': {'elevation': {'band_step_feet': 1000}}},
        )


@pytest.mark.parametrize(
    ('block', 'expected'),
    [
        ({'band_step_ft': 1000}, BandStepParams(band_step_ft=1000)),
        ({'buckets': 4}, BucketParams(buckets=4)),
        ({'threshold_pct': 50}, ThresholdParams(threshold_pct=50)),
        ({'entropy_threshold': 0.5}, EntropyThresholdParams(entropy_threshold=0.5)),
        (None, None),
    ],
)
def test_zone_param_block_parses_to_its_member_model(block, expected):
    # The single param field name routes an on-disk block to exactly one member
    # model; null means "enabled, no params" (a categorical axis).
    config = DatasetConfig(
        grid=_grid_config(),
        variables={'swe': _swe_variable()},
        zones={'terrain': {'layer': block}},
    )
    assert config.zones['terrain']['layer'] == expected


def test_zone_param_block_with_params_of_two_kinds_is_rejected():
    # extra='forbid' on every member keeps the union disjoint: a block carrying
    # two different schemes' params matches no member.
    with pytest.raises(ValidationError):
        DatasetConfig(
            grid=_grid_config(),
            variables={'swe': _swe_variable()},
            zones={'terrain': {'elevation': {'band_step_ft': 1000, 'buckets': 4}}},
        )


def test_zone_params_serialize_as_their_single_field():
    config = DatasetConfig(
        grid=_grid_config(),
        variables={'swe': _swe_variable()},
        zones={
            'terrain': {'elevation': BandStepParams(band_step_ft=1000), 'aspect': None},
        },
    )
    text = config.model_dump_json()
    assert '"elevation":{"band_step_ft":1000}' in text
    assert '"aspect":null' in text


def test_default_zones_enumerate_every_served_layer():
    # DEFAULT_ZONES pins the full set of served layers with their
    # behaviour-preserving defaults: terrain's five (aspect is categorical, so no
    # param) and land cover's forest_cover.
    assert {
        'terrain': {
            'elevation': BandStepParams(band_step_ft=1000),
            'aspect': None,
            'northness': BucketParams(buckets=4),
            'eastness': BucketParams(buckets=4),
            'aspect_entropy': EntropyThresholdParams(entropy_threshold=0.5),
        },
        'landcover': {'forest_cover': ThresholdParams(threshold_pct=50)},
    } == DEFAULT_ZONES


def test_default_zones_round_trip_through_dataset_config(tmp_path):
    # The full DEFAULT_ZONES survives save -> load_entity unchanged.
    config = DatasetConfig(
        grid=_grid_config(),
        variables={'swe': _swe_variable()},
        zones=DEFAULT_ZONES,
    )
    path = tmp_path / 'dataset.json'
    config.save(path)

    loaded = load_entity(path)
    assert isinstance(loaded, DatasetConfig)
    assert loaded.zones == DEFAULT_ZONES


def test_footprint_round_trips_through_json(tmp_path):
    import shapely

    from geojson_pydantic.geometries import Geometry
    from pydantic import TypeAdapter

    footprint = TypeAdapter(Geometry).validate_python(
        shapely.geometry.mapping(shapely.box(-119.5, 44.0, -119.0, 44.5)),
    )
    config = DatasetConfig(
        grid=_grid_config(),
        variables={'swe': _swe_variable()},
        footprint=footprint,
    )
    path = tmp_path / 'dataset.json'
    config.save(path)

    reloaded = DatasetConfig.load(path)
    assert reloaded.footprint is not None
    assert reloaded.footprint == footprint
