"""Shared dataset-selection helpers and common options for the CLI groups.

The ``dataset`` and ``report`` groups both resolve datasets by name (or default
to all configured) and share the ``--format`` / ``--dataset`` options; those live
here so a command body stays a thin wrapper over the domain API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from snowtool.cli._render import FORMATS

if TYPE_CHECKING:
    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.db import SnowDb
    from snowtool.snowdb.manager import SnowDbManager

format_option = click.option(
    '--format',
    'fmt',
    type=click.Choice(FORMATS),
    default='table',
    help='Output format.',
)

dataset_option = click.option(
    '--dataset',
    '-d',
    'dataset_names',
    multiple=True,
    help='Dataset to act on (repeatable; default: all configured datasets).',
)


def get_dataset(snowdb: SnowDb, name: str) -> Dataset:
    """Resolve a dataset by name, or raise a clean CLI error listing the options."""
    try:
        return snowdb[name]
    except KeyError as e:
        configured = ', '.join(sorted(snowdb)) or '(none)'
        raise click.ClickException(
            f'No such dataset: {name!r}. Configured datasets: {configured}.',
        ) from e


def resolve_datasets(snowdb: SnowDb, names: tuple[str, ...]) -> list[Dataset]:
    """The datasets named by ``-d`` (each validated), or all configured if none."""
    if not names:
        return [snowdb[name] for name in sorted(snowdb)]
    return [get_dataset(snowdb, name) for name in names]


def require_initialized(manager: SnowDbManager) -> None:
    """Refuse to run a write command against an un-initialized snowdb root."""
    try:
        manager.require_initialized()
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
