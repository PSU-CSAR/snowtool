"""Shared CLI rendering: the ``--format`` emitter.

Commands compute plain rows (lists of dicts / dumped pydantic models) on the
domain side and hand them to :func:`_emit`, which is the only place output
formatting lives -- so every command renders ``table``/``json``/``csv``
identically.
"""

from __future__ import annotations

import csv
import io
import json

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import click

from rich import box
from rich.table import Table

from snowtool.cli import _console

if TYPE_CHECKING:
    from collections.abc import Iterable

# The shared --format choices; commands reuse this for their option.
FORMATS = ('table', 'json', 'csv')


def _emit(rows: Iterable[Mapping[str, Any]], fmt: str = 'table') -> None:
    """Render ``rows`` (uniform string-keyed mappings) to stdout in ``fmt``.

    ``json`` always emits (an empty list as ``[]``); ``table``/``csv`` use the
    first row's keys as the column order and emit nothing for an empty result.
    """
    materialized = [dict(row) for row in rows]

    if fmt == 'json':
        click.echo(json.dumps(materialized, default=str, indent=2))
        return

    if not materialized:
        return

    headers = list(materialized[0].keys())

    if fmt == 'csv':
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=headers)
        writer.writeheader()
        writer.writerows(
            {key: _scalar(value) for key, value in row.items()} for row in materialized
        )
        click.echo(buffer.getvalue(), nl=False)
        return

    table = Table(box=box.SIMPLE_HEAD, header_style='bold', pad_edge=False)
    for header in headers:
        table.add_column(header, overflow='fold')
    for row in materialized:
        table.add_row(*(_scalar(row.get(header, '')) for header in headers))
    _console.out().print(table)


def _scalar(value: Any) -> str:
    """Flatten a value for table/csv cells.

    Lists/tuples become comma-joined; mappings (e.g. a pourpoint's per-dataset
    coverage) become comma-joined ``key=value`` pairs -- either way avoiding a raw
    Python repr in a table/csv cell.
    """
    if isinstance(value, Mapping):
        return ', '.join(f'{key}={item}' for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return ', '.join(str(item) for item in value)
    return str(value)


def _emit_record(record: Mapping[str, Any], fmt: str = 'table') -> None:
    """Render a single record (one entity, e.g. ``dataset info``) in ``fmt``.

    ``json`` dumps the mapping as-is (lists preserved); ``table`` prints a
    borderless key/value table; ``csv`` writes a header row + one value row.
    List values are comma-joined for the table/csv (non-json) forms.
    """
    record = dict(record)

    if fmt == 'json':
        click.echo(json.dumps(record, default=str, indent=2))
        return

    if fmt == 'csv':
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(record))
        writer.writeheader()
        writer.writerow({key: _scalar(value) for key, value in record.items()})
        click.echo(buffer.getvalue(), nl=False)
        return

    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column(style='bold')
    table.add_column(overflow='fold')
    for key, value in record.items():
        table.add_row(key, _scalar(value))
    _console.out().print(table)
