"""The typed dataset config: byte-equal round-trips, the union, ingester resolve."""

import pytest

from pydantic import ValidationError

from snowtool.snowdb.config import (
    DatasetConfig,
    GridConfig,
    RootConfig,
    UnitConfig,
    VariableConfig,
    load_entity,
)
from snowtool.snowdb.datasets import (
    DATASET_TEMPLATES,
    DEFAULT_DATASET_SPECS,
    config_from_spec,
)
from snowtool.snowdb.spec import DatasetSpec
from snowtool.snowdb.variables import Reducer


def _grid_config() -> GridConfig:
    return GridConfig(
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
        # Exact geometric *and* coordinate equality (the GeoJSON round-trip must
        # not perturb the served footprint at all).
        assert resolved.footprint is not None
        assert resolved.footprint.equals_exact(spec.footprint, 0)


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
    from snowtool.snowdb.datasets import SwannIngester

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
            'swe': VariableConfig(
                unit=UnitConfig(name='mm', scale_factor=1),
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
