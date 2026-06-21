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


@pytest.mark.parametrize('value', ['20180427', '2018-04-27'])
def test_date_param_accepts_both_formats(value):
    assert DATE.convert(value, None, None) == date(2018, 4, 27)


def test_date_param_passes_through_date():
    assert DATE.convert(date(2018, 4, 27), None, None) == date(2018, 4, 27)


def test_date_param_rejects_garbage():
    with pytest.raises(click.BadParameter):
        DATE.convert('not-a-date', None, None)
