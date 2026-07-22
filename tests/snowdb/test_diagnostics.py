"""Date-gap diagnostics (pure) and the dataset_status scan builder."""

from datetime import date

import pytest

from snowtool.snowdb.diagnostics import dataset_status, date_gaps


def d(day: int) -> date:
    return date(2018, 4, day)


@pytest.mark.parametrize(
    ('dates', 'expected'),
    [
        ([], []),
        ([d(1)], []),
        ([d(1), d(2), d(3)], []),
        ([d(1), d(3)], [(d(2), d(2))]),
        ([d(1), d(5)], [(d(2), d(4))]),
        ([d(1), d(3), d(4), d(8)], [(d(2), d(2)), (d(5), d(7))]),
        ([d(8), d(1), d(3), d(3), d(4)], [(d(2), d(2)), (d(5), d(7))]),
        # Nothing before the first or after the last date is a "gap".
        ([d(2), d(3), d(4)], []),
        (
            [date(2018, 1, 30), date(2018, 2, 2)],
            [(date(2018, 1, 31), date(2018, 2, 1))],
        ),
    ],
    ids=[
        'no_dates',
        'single_date',
        'contiguous',
        'single_missing_day',
        'multi_day_run',
        'several_gaps_in_order',
        'unordered_and_duplicate',
        'only_interior_gaps',
        'month_boundary',
    ],
)
def test_date_gaps(dates, expected):
    assert date_gaps(dates) == expected


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
