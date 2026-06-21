"""Date-gap diagnostics (pure) and the dataset_status scan builder."""

from datetime import date

import pytest

from snowtool.snowdb.diagnostics import dataset_status, date_gaps


def d(day: int) -> date:
    return date(2018, 4, day)


def test_no_dates_has_no_gaps():
    assert date_gaps([]) == []


def test_single_date_has_no_gaps():
    assert date_gaps([d(1)]) == []


def test_contiguous_dates_have_no_gaps():
    assert date_gaps([d(1), d(2), d(3)]) == []


def test_single_missing_day_is_a_one_day_gap():
    assert date_gaps([d(1), d(3)]) == [(d(2), d(2))]


def test_multi_day_gap_is_inclusive_run():
    assert date_gaps([d(1), d(5)]) == [(d(2), d(4))]


def test_several_gaps_reported_in_order():
    assert date_gaps([d(1), d(3), d(4), d(8)]) == [(d(2), d(2)), (d(5), d(7))]


def test_unordered_and_duplicate_dates_are_normalized():
    assert date_gaps([d(8), d(1), d(3), d(3), d(4)]) == [(d(2), d(2)), (d(5), d(7))]


def test_only_interior_gaps_reported():
    # Nothing before the first or after the last date is a "gap".
    span = [d(2), d(3), d(4)]
    assert date_gaps(span) == []


def test_gap_spanning_a_month_boundary():
    assert date_gaps([date(2018, 1, 30), date(2018, 2, 2)]) == [
        (date(2018, 1, 31), date(2018, 2, 1)),
    ]


@pytest.mark.parametrize(
    ('dates', 'expected'),
    [
        ([d(1), d(2)], []),
        ([d(1), d(4)], [(d(2), d(3))]),
    ],
)
def test_parametrized_small_cases(dates, expected):
    assert date_gaps(dates) == expected


def test_dataset_status_for_created_dataset(dataset):
    status = dataset_status(dataset)

    assert status.name == 'test'
    assert status.present is True
    assert status.artifacts.dem is True
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
