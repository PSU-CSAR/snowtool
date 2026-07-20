"""Shared dataset-selection helpers and common options for the CLI groups.

The ``dataset``, ``doctor``, and ``pourpoint`` groups all resolve datasets by
name (or default to a whole-database sweep) and share the ``--format`` /
``--dataset`` options; those live here so a command body stays a thin wrapper
over the domain API. The read surface (``dataset dates``/``values``/``info``)
resolves everything registered (active or not); the API is what restricts
readers to active datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from snowtool.cli._render import FORMATS
from snowtool.exceptions import UnknownDatasetError

if TYPE_CHECKING:
    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.db import SnowDb

format_option = click.option(
    '--format',
    'fmt',
    type=click.Choice(FORMATS),
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

dataset_option = click.option(
    '--dataset',
    '-d',
    'dataset_names',
    multiple=True,
    help='Dataset to act on, active or not (repeatable; default: every '
    'registered dataset -- `doctor` narrows this to active unless '
    '--include-inactive).',
)


def get_dataset(
    snowdb: SnowDb,
    name: str,
    *,
    include_inactive: bool = True,
) -> Dataset:
    """Resolve a dataset by name, or raise a clean CLI error listing the options.

    By default anything *registered* resolves -- the management/diagnostics
    surface, where activation is irrelevant. ``include_inactive=False`` narrows
    the lookup to active datasets (matching what the API serves; ``stats`` is
    the one reader-surface caller); a registered-but-inactive name then gets a
    pointed "activate it" error instead of a generic miss.
    """
    if include_inactive and name in snowdb.registered:
        return snowdb.registered[name]
    try:
        return snowdb[name]
    except UnknownDatasetError as e:
        if name in snowdb.registered:
            raise click.ClickException(
                f'Dataset {name!r} is registered but inactive. '
                f"Activate it with 'snowtool dataset activate {name}'.",
            ) from e
        registered = ', '.join(sorted(snowdb.registered)) or '(none)'
        raise click.ClickException(
            f'No such dataset: {name!r}. Registered datasets: {registered}.',
        ) from e


def resolve_datasets(
    snowdb: SnowDb,
    names: tuple[str, ...],
    *,
    include_inactive: bool = True,
) -> list[Dataset]:
    """The datasets named by ``-d`` (each validated), or a whole-database default.

    An explicit ``-d`` NAME always resolves from everything registered (naming
    an inactive dataset directly means you want it acted on). With no names the
    default pool is every *registered* dataset -- the diagnostics/management
    surface (reports, AOI rasterization) treats active and inactive datasets
    identically -- unless ``include_inactive=False`` narrows it to the active
    set (``doctor``'s default, so a half-built staged dataset does not
    fail the CI gate).
    """
    if not names:
        pool = snowdb.registered if include_inactive else snowdb.datasets
        return [pool[name] for name in sorted(pool)]
    return [get_dataset(snowdb, name) for name in names]
