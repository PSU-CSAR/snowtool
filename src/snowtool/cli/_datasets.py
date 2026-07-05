"""Shared dataset-selection helpers and common options for the CLI groups.

The ``dataset`` and ``report`` groups both resolve datasets by name (or default
to a whole-database sweep) and share the ``--format`` / ``--dataset`` options;
those live here so a command body stays a thin wrapper over the domain API. The
read surface (query) resolves *active* datasets only; the management/diagnostics
surface resolves everything registered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from snowtool.cli._render import FORMATS

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

# The same --format flag for commands whose output is nested (e.g. `query stats`):
# there is no table form, so the choice is restricted to the two flat serializations.
nested_format_option = click.option(
    '--format',
    'fmt',
    type=click.Choice(('csv', 'json')),
    default='csv',
    help='Output format (the zonal output is nested, so no table form).',
)

dataset_option = click.option(
    '--dataset',
    '-d',
    'dataset_names',
    multiple=True,
    help='Dataset to act on, active or not (repeatable; default: every '
    'registered dataset, except `snowdb validate` which defaults to active).',
)


def get_dataset(
    snowdb: SnowDb,
    name: str,
    *,
    include_inactive: bool = False,
) -> Dataset:
    """Resolve a dataset by name, or raise a clean CLI error listing the options.

    By default only *active* datasets resolve (the read surface: query/report
    serve what readers serve); a registered-but-inactive name gets a pointed
    "activate it" error instead of a generic miss. ``include_inactive`` widens
    the lookup to everything registered -- the management/diagnostics surface,
    where activation is irrelevant.
    """
    if include_inactive and name in snowdb.registered:
        return snowdb.registered[name]
    try:
        return snowdb[name]
    except KeyError as e:
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
    set (``snowdb validate``'s default, so a half-built staged dataset does not
    fail the CI gate).
    """
    if not names:
        pool = snowdb.registered if include_inactive else snowdb.datasets
        return [pool[name] for name in sorted(pool)]
    return [get_dataset(snowdb, name, include_inactive=True) for name in names]
