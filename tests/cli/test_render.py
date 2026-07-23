"""The shared --format emitter."""

import json

from snowtool.cli._render import emit

ROWS = [
    {'name': 'snodas', 'dates': 3},
    {'name': 'test', 'dates': 10},
]


def test_emit_json_roundtrips(capsys):
    emit(ROWS, 'json')

    assert json.loads(capsys.readouterr().out) == ROWS


def test_emit_json_empty_is_empty_list(capsys):
    emit([], 'json')

    assert json.loads(capsys.readouterr().out) == []


def test_emit_csv_has_header_and_rows(capsys):
    emit(ROWS, 'csv')
    lines = capsys.readouterr().out.splitlines()

    assert lines[0] == 'name,dates'
    assert lines[1] == 'snodas,3'
    assert lines[2] == 'test,10'


def test_emit_table_has_header_and_rows(capsys):
    emit(ROWS, 'table')
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]

    header, *rest = lines
    assert 'name' in header
    assert 'dates' in header
    data = [ln for ln in rest if 'snodas' in ln or ln.strip().startswith('test')]
    assert len(data) == 2
    assert 'snodas' in data[0]
    assert '3' in data[0]


def test_emit_record_table_is_key_value(capsys):
    from snowtool.cli._render import emit_record

    emit_record({'name': 'snodas', 'dates': 3}, 'table')
    out = capsys.readouterr().out
    assert 'name' in out
    assert 'snodas' in out
    assert 'dates' in out
    assert '3' in out


def test_emit_table_empty_prints_nothing(capsys):
    emit([], 'table')

    assert capsys.readouterr().out == ''


def test_emit_table_does_not_wrap_wide_rows_on_non_terminal(capsys):
    # pytest capture is non-terminal, matching piped/redirected CI output --
    # exactly where a wrapped value (e.g. '20240101' -> '20240\n101') would
    # break parsing. Long values across many columns must survive intact.
    row = {f'col{i:02d}': f'value-{i:02d}-abcdefghijklmnop' for i in range(12)}
    emit([row], 'table')
    output = capsys.readouterr().out

    for value in row.values():
        assert value in output


def test_info_table_record_renders_geographic_and_elevation_prose():
    # The `dataset info` table form alone substitutes prose for the typed fields:
    # a geographic grid's null cell area becomes 'varies (geographic)', and the
    # two numeric elevation fields collapse to a 'MIN .. MAX' bracket. json/csv
    # keep the typed fields (cell_area_m2=None, numeric min/max), pinned in
    # test_dataset_cli.py. This lives here with the other output-format cases.
    from snowtool.cli.dataset import _info_table_record
    from snowtool.snowdb.dataset import DatasetArtifacts
    from snowtool.snowdb.diagnostics import (
        DatasetInfoReport,
        DatasetStatus,
        GridReport,
    )

    report = DatasetInfoReport(
        name='test',
        active=True,
        status=DatasetStatus(
            name='test',
            present=True,
            artifacts=DatasetArtifacts(zone_layers={}, aoi_rasters=True, cogs=True),
            date_count=0,
            first_date=None,
            last_date=None,
        ),
        grid=GridReport(
            name='test',
            crs='EPSG:4326',
            is_geographic=True,
            rows=512,
            cols=512,
            px_size=0.01,
            tile_size=256,
            n_tiles=4,
            extent=(-120.0, 44.0, -119.0, 45.0),
            cell_area_m2=None,  # geographic -> no fixed per-cell area
        ),
        zones={},
        min_elevation_m=-100.0,
        max_elevation_m=4500.0,
        variables=('swe',),
        zone_layers={},
    )

    record = _info_table_record(report)

    assert record['cell_area_m2'] == 'varies (geographic)'
    assert record['elevation_bracket_m'] == '-100.0 .. 4500.0'
    assert 'min_elevation_m' not in record
    assert 'max_elevation_m' not in record
