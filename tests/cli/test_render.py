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


def test_emit_table_has_header_and_rows(capsys):
    _emit(ROWS, 'table')
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]

    header, *rest = lines
    assert 'name' in header
    assert 'dates' in header
    data = [ln for ln in rest if 'snodas' in ln or ln.strip().startswith('test')]
    assert len(data) == 2
    assert 'snodas' in data[0]
    assert '3' in data[0]


def test_emit_record_table_is_key_value(capsys):
    from snowtool.cli._render import _emit_record

    _emit_record({'name': 'snodas', 'dates': 3}, 'table')
    out = capsys.readouterr().out
    assert 'name' in out
    assert 'snodas' in out
    assert 'dates' in out
    assert '3' in out


def test_emit_table_empty_prints_nothing(capsys):
    _emit([], 'table')

    assert capsys.readouterr().out == ''


def test_emit_table_does_not_wrap_wide_rows_on_non_terminal(capsys):
    # pytest capture is non-terminal, matching piped/redirected CI output --
    # exactly where a wrapped value (e.g. '20240101' -> '20240\n101') would
    # break parsing. Long values across many columns must survive intact.
    row = {f'col{i:02d}': f'value-{i:02d}-abcdefghijklmnop' for i in range(12)}
    _emit([row], 'table')
    output = capsys.readouterr().out

    for value in row.values():
        assert value in output


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
