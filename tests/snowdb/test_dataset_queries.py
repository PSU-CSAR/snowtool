"""Read-only Dataset query helpers backing the report/diagnostics commands."""

from datetime import date

from snowtool.snowdb.dataset import Dataset, DatasetArtifacts
from snowtool.snowdb.pourpoint import Pourpoint
from snowtool.snowdb.spec import DatasetSpec, GridParams
from snowtool.snowdb.terrain import ELEVATION


def test_available_dates_empty_when_no_cogs(dataset):
    # A freshly created dataset has an (empty) cogs/ dir and no date dirs.
    assert dataset.available_dates() == []


def test_available_dates_lists_and_sorts_date_dirs(dataset):
    for name in ('20180427', '20180101', '20180415'):
        (dataset._cogs / name).mkdir()
    # Non-date / non-dir entries are ignored.
    (dataset._cogs / 'not-a-date').mkdir()
    (dataset._cogs / 'stray.txt').write_text('x')

    assert dataset.available_dates() == [
        date(2018, 1, 1),
        date(2018, 4, 15),
        date(2018, 4, 27),
    ]


def test_available_dates_picks_up_ingested_cog(dataset, swe_cog):
    assert dataset.available_dates() == [date(2018, 4, 27)]


def test_date_dir_points_at_the_cogs_subdir(dataset):
    assert dataset.date_dir(date(2018, 4, 27)) == dataset._cogs / '20180427'


def test_missing_variables_absent_date_is_all_variables(dataset):
    missing = dataset.missing_variables(date(2018, 4, 27))
    assert {v.key for v in missing} == set(dataset.spec.variables)


def test_missing_variables_excludes_present_one(dataset, swe_cog):
    # swe_cog provides only the SWE product for 2018-04-27.
    missing = dataset.missing_variables(date(2018, 4, 27))
    missing_keys = {v.key for v in missing}

    assert 'swe' not in missing_keys
    assert missing_keys == set(dataset.spec.variables) - {'swe'}


def test_aoi_raster_triplets_empty_by_default(dataset):
    assert dataset.aoi_raster_triplets() == set()


def test_aoi_raster_triplets_from_burned_rasters(dataset, pourpoint_geojson):
    dataset.rasterize_aoi(Pourpoint.from_geojson(pourpoint_geojson))

    assert dataset.aoi_raster_triplets() == {'12345:MT:USGS'}


def test_artifact_status_full_for_created_geographic_dataset(dataset):
    status = dataset.artifact_status()

    assert status == DatasetArtifacts(
        zone_layers={'terrain': True, 'landcover': True},
        aoi_rasters=True,
        cogs=True,
    )


def test_artifact_status_reports_missing_terrain(dataset):

    dataset.zones['terrain'].layer_path(ELEVATION).unlink()

    assert dataset.artifact_status().zone_layers['terrain'] is False


def test_dates_before_filters_strictly(dataset):
    for name in ('20180101', '20180201', '20180301'):
        (dataset._cogs / name).mkdir()

    assert dataset.dates_before(date(2018, 2, 1)) == [date(2018, 1, 1)]
    assert dataset.dates_before(date(2018, 4, 1)) == [
        date(2018, 1, 1),
        date(2018, 2, 1),
        date(2018, 3, 1),
    ]


def test_remove_date_deletes_and_reports(dataset):
    (dataset._cogs / '20180101').mkdir()

    assert dataset.remove_date(date(2018, 1, 1)) is True
    assert not (dataset._cogs / '20180101').exists()


def test_remove_absent_date_is_a_noop(dataset):
    assert dataset.remove_date(date(2018, 1, 1)) is False


def test_artifact_status_for_created_projected_dataset(tmp_path):
    # No area raster is tracked for any grid -- the AOI raster carries cell area.

    spec = DatasetSpec(
        name='utm',
        grid_params=GridParams(
            origin_x=500000.0,
            origin_y=5000000.0,
            px_size=30.0,
            cols=256,
            rows=256,
            tile_size=256,
            crs=32611,
        ),
    )
    dataset = Dataset.create(spec, tmp_path / 'utm')

    status = dataset.artifact_status()
    assert status.aoi_rasters is True
    assert status.cogs is True
    assert not hasattr(status, 'area')
