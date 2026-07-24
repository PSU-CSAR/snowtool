"""Shared CLI rendering: the ``--format`` option and emitter.

Commands compute plain rows (lists of dicts / dumped pydantic models) on the
domain side and hand them to :func:`emit`, which is the only place output
formatting lives -- so every command renders ``table``/``json``/``csv``
identically. The ``--format`` option decorators (:data:`format_option`,
:data:`nested_format_option`) live here so a command imports its option and
emitter from one place.
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

format_option = click.option(
    '--format',
    'fmt',
    type=click.Choice(('table', 'json', 'csv')),
    default='table',
    help='Output format.',
)

# The same --format flag for commands whose output is nested (e.g. `stats`):
# there is no table form, so the choice is the two flat serializations. ``json``
# is the compact/normalized stats body (see ZonalStats.dump_compact).
nested_format_option = click.option(
    '--format',
    'fmt',
    type=click.Choice(('csv', 'json')),
    default='json',
    help='Output format (json = compact stats body; csv = flat rows).',
)


def emit(rows: Iterable[Mapping[str, Any]], fmt: str = 'table') -> None:
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


def emit_record(record: Mapping[str, Any], fmt: str = 'table') -> None:
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
        emit([record], fmt)
        return

    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column(style='bold')
    table.add_column(overflow='fold')
    for key, value in record.items():
        table.add_row(key, _scalar(value))
    _console.out().print(table)
