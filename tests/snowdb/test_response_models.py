"""The per-dataset zonal-stats response models generated from a DatasetSpec."""

import math

from datetime import date

import pytest

from pydantic import ValidationError

from snowtool.snowdb.datasets import SNODAS_SPEC


def test_model_has_a_field_per_variable(spec):
    fields = spec.zonal_stat_model.model_fields
    assert 'min_elevation_ft' in fields
    assert 'max_elevation_ft' in fields
    assert 'area_m2' in fields
    for variable in spec.variables.values():
        assert variable.stat_name in fields


def test_models_are_cached_on_the_spec(spec):
    assert spec.zonal_stat_model is spec.zonal_stat_model
    assert spec.zonal_stats_model is spec.zonal_stats_model


def test_model_names_are_namespaced_by_dataset():
    assert SNODAS_SPEC.zonal_stat_model.__name__ == 'SnodasZonalStat'
    assert SNODAS_SPEC.zonal_stats_model.__name__ == 'SnodasZonalStats'


def test_nan_stat_serializes_to_null_but_stays_nan_in_memory(spec):
    zone = spec.zonal_stat_model(
        min_elevation_ft=0,
        max_elevation_ft=1000,
        area_m2=0.0,
        mean_swe_mm=float('nan'),
    )
    # The in-memory value is still nan (zonal_stats relies on this)...
    assert math.isnan(zone.mean_swe_mm)
    # ...but it serializes to null, so the JSON is valid (no NaN literal).
    assert zone.model_dump(mode='json')['mean_swe_mm'] is None
    assert 'NaN' not in zone.model_dump_json()


def test_real_stat_value_passes_through(spec):
    zone = spec.zonal_stat_model(
        min_elevation_ft=0,
        max_elevation_ft=1000,
        area_m2=5.0,
        mean_swe_mm=12.5,
    )
    dumped = zone.model_dump(mode='json')
    assert dumped['mean_swe_mm'] == 12.5
    assert dumped['area_m2'] == 5.0


def test_area_must_be_non_negative(spec):
    with pytest.raises(ValidationError):
        spec.zonal_stat_model(
            min_elevation_ft=0,
            max_elevation_ft=1000,
            area_m2=-1.0,
        )


def test_stats_model_wraps_a_date_and_its_zones(spec):
    zone = spec.zonal_stat_model(
        min_elevation_ft=0,
        max_elevation_ft=1000,
        area_m2=0.0,
    )
    stats = spec.zonal_stats_model(date=date(2018, 4, 27), zones=[zone])
    assert stats.date == date(2018, 4, 27)
    assert len(stats.zones) == 1
