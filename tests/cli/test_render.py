"""The shared --format emitter."""

import json

from snowtool.cli._render import _emit

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
