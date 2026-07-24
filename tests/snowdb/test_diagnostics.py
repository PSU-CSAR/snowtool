"""The dataset_status scan builder."""

from datetime import date

from snowtool.snowdb.diagnostics import dataset_status


def test_dataset_status_for_created_dataset(dataset):
    status = dataset_status(dataset)

    assert status.name == 'test'
    assert status.present is True
    assert status.artifacts.zone_layers['terrain'] is True
    assert status.date_count == 0
    assert status.first_date is None
    assert status.last_date is None


def test_dataset_status_with_dates(dataset):
    for name in ('20180101', '20180301'):
        (dataset._cogs / name).mkdir()

    status = dataset_status(dataset)

    assert status.date_count == 2
    assert status.first_date == date(2018, 1, 1)
    assert status.last_date == date(2018, 3, 1)
