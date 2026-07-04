"""The shared --format emitter and the DATE argument type."""

import json

from datetime import date

import click
import pytest

from snowtool.cli._render import DATE, _emit

ROWS = [
    {'name': 'snodas', 'dates': 3},
    {'name': 'test', 'dates': 10},
]


def test_emit_json_roundtrips(capsys):
    _emit(ROWS, 'json')

    assert json.loads(capsys.readouterr().out) == ROWS


def test_emit_json_empty_is_empty_list(capsys):
    _emit([], 'json')

    assert json.loads(capsys.readouterr().out) == []


def test_emit_csv_has_header_and_rows(capsys):
    _emit(ROWS, 'csv')
    lines = capsys.readouterr().out.splitlines()

    assert lines[0] == 'name,dates'
    assert lines[1] == 'snodas,3'
    assert lines[2] == 'test,10'


def test_emit_table_aligns_columns(capsys):
    _emit(ROWS, 'table')
    lines = capsys.readouterr().out.splitlines()

    # Header then one line per row; the name column is padded to its widest cell.
    assert lines[0].startswith('name')
    assert 'dates' in lines[0]
    assert lines[1].startswith('snodas')
    assert lines[2].startswith('test  ')  # padded to width of 'snodas'


def test_emit_table_empty_prints_nothing(capsys):
    _emit([], 'table')

    assert capsys.readouterr().out == ''


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ('20180427', date(2018, 4, 27)),
        ('2018-04-27', date(2018, 4, 27)),
        # Timezone-independence regression (2e935b6): to_date must take .date()
        # straight off the parsed datetime, not reinterpret it via astimezone,
        # which shifted the result across the local-TZ boundary (e.g. '20240101'
        # -> 2023-12-31 under TZ=Asia/Tokyo).
        ('20240101', date(2024, 1, 1)),
        ('2024-01-01', date(2024, 1, 1)),
        ('20240229', date(2024, 2, 29)),  # leap day
        ('2024-02-29', date(2024, 2, 29)),  # leap day, dashed
        ('20231231', date(2023, 12, 31)),
    ],
)
def test_date_param_parses_exact_value(value, expected):
    assert DATE.convert(value, None, None) == expected


def test_date_param_passes_through_date():
    assert DATE.convert(date(2018, 4, 27), None, None) == date(2018, 4, 27)


@pytest.mark.parametrize(
    'value',
    [
        'not-a-date',
        '2024-02-30',  # not a leap day
        '20240230',
        '',
        '2024/01/01',
    ],
)
def test_date_param_rejects_invalid_input(value):
    with pytest.raises(click.BadParameter):
        DATE.convert(value, None, None)
