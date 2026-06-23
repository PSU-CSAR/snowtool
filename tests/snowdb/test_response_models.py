"""The per-dataset zonal-stats response models generated from a DatasetSpec."""

import math

from datetime import date

import pytest

from pydantic import ValidationError

from snowtool.snowdb.datasets import SNODAS_SPEC


def _band_ref(min_ft: int = 0, max_ft: int = 1000) -> dict:
    return {
        'kind': 'band',
        'layer': 'terrain.elevation',
        'min': min_ft,
        'max': max_ft,
        'unit': 'ft',
    }


def test_model_has_zone_and_a_field_per_variable(spec):
    fields = spec.zonal_stat_model.model_fields
    assert 'zone' in fields
    assert 'area_m2' in fields
    # The old hardcoded elevation fields are gone -- the band is in the zone refs.
    assert 'min_elevation_ft' not in fields
    assert 'max_elevation_ft' not in fields
    for variable in spec.variables.values():
        assert variable.stat_name in fields


def test_models_are_cached_on_the_spec(spec):
    assert spec.zonal_stat_model is spec.zonal_stat_model
    assert spec.zonal_stats_model is spec.zonal_stats_model


def test_model_names_are_namespaced_by_dataset():
    assert SNODAS_SPEC.zonal_stat_model.__name__ == 'SnodasZonalStat'
    assert SNODAS_SPEC.zonal_stats_model.__name__ == 'SnodasZonalStats'


def test_zone_refs_discriminate_band_class_and_threshold(spec):
    cell = spec.zonal_stat_model(
        zone=[
            _band_ref(3000, 4000),
            {'kind': 'class', 'layer': 'terrain.aspect', 'code': 0, 'label': 'N'},
            {
                'kind': 'threshold',
                'layer': 'landcover.forest_cover',
                'threshold': 50,
                'unit': '%',
                'side': 'above',
                'label': 'forested',
            },
        ],
        area_m2=10.0,
    )
    band, klass, thresh = cell.zone
    assert (band.min, band.max, band.unit) == (3000, 4000, 'ft')
    assert (klass.code, klass.label) == (0, 'N')
    assert (thresh.threshold, thresh.unit, thresh.side, thresh.label) == (
        50,
        '%',
        'above',
        'forested',
    )


def test_nan_stat_serializes_to_null_but_stays_nan_in_memory(spec):
    cell = spec.zonal_stat_model(
        zone=[_band_ref()],
        area_m2=0.0,
        mean_swe_mm=float('nan'),
    )
    # The in-memory value is still nan (zonal_stats relies on this)...
    assert math.isnan(cell.mean_swe_mm)
    # ...but it serializes to null, so the JSON is valid (no NaN literal).
    assert cell.model_dump(mode='json')['mean_swe_mm'] is None
    assert 'NaN' not in cell.model_dump_json()


def test_real_stat_value_passes_through(spec):
    cell = spec.zonal_stat_model(
        zone=[_band_ref()],
        area_m2=5.0,
        mean_swe_mm=12.5,
    )
    dumped = cell.model_dump(mode='json')
    assert dumped['mean_swe_mm'] == 12.5
    assert dumped['area_m2'] == 5.0
    assert dumped['zone'][0]['min'] == 0


def test_area_must_be_non_negative(spec):
    with pytest.raises(ValidationError):
        spec.zonal_stat_model(zone=[_band_ref()], area_m2=-1.0)


def test_stats_model_wraps_a_date_zone_layers_and_its_cells(spec):
    cell = spec.zonal_stat_model(zone=[_band_ref()], area_m2=0.0)
    stats = spec.zonal_stats_model(
        date=date(2018, 4, 27),
        zone_layers=['terrain.elevation'],
        zones=[cell],
    )
    assert stats.date == date(2018, 4, 27)
    assert stats.zone_layers == ['terrain.elevation']
    assert len(stats.zones) == 1
