"""Shared CLI rendering: the ``--format`` emitter and a date argument type.

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
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from collections.abc import Iterable

# The shared --format choices; commands reuse this for their option.
FORMATS = ('table', 'json', 'csv')


def _to_date(value: str) -> date:
    """Parse ``YYYYMMDD``/``YYYY-MM-DD`` into a :class:`date`, timezone-independent.

    Takes ``.date()`` straight off the parsed naive datetime rather than
    reinterpreting it via ``astimezone``, which would shift the result across the
    local-TZ boundary (e.g. ``'20240101'`` -> 2023-12-31 under ``TZ=Asia/Tokyo``).
    """
    return datetime.strptime(  # noqa: DTZ007
        value.replace('-', ''),
        '%Y%m%d',
    ).date()


class DateParamType(click.ParamType):
    """A click argument/option type accepting ``YYYYMMDD`` or ``YYYY-MM-DD``."""

    name = 'date'

    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> date:
        if isinstance(value, date):
            return value
        try:
            return _to_date(value)
        except ValueError:
            self.fail(
                f'{value!r} is not a valid date (expected YYYYMMDD or YYYY-MM-DD)',
                param,
                ctx,
            )


# A reusable instance for command parameter declarations.
DATE = DateParamType()


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

    def cells(row: Mapping[str, Any]) -> list[str]:
        return [_scalar(row.get(header, '')) for header in headers]

    rendered = [headers, *(cells(row) for row in materialized)]
    widths = [max(len(row[col]) for row in rendered) for col in range(len(headers))]
    for row in rendered:
        click.echo('  '.join(cell.ljust(widths[col]) for col, cell in enumerate(row)))


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

    ``json`` dumps the mapping as-is (lists preserved); ``table`` prints aligned
    ``key  value`` lines; ``csv`` writes a header row + one value row. List
    values are comma-joined for the table/csv (non-json) forms.
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

    width = max((len(key) for key in record), default=0)
    for key, value in record.items():
        click.echo(f'{key.ljust(width)}  {_scalar(value)}')
