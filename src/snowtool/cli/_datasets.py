"""Shared dataset-selection helpers and common options for the CLI groups.

The ``dataset``, ``doctor``, and ``pourpoint`` groups all resolve datasets by
name (or default to a whole-database sweep) and share the ``--dataset`` option;
it lives here so a command body stays a thin wrapper over the domain API (the
``--format`` option lives with its emitter in :mod:`snowtool.cli._render`). The
read surface (``dataset dates``/``values``/``info``)
resolves everything registered (active or not); the API is what restricts
readers to active datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.db import SnowDb

dataset_option = click.option(
    '--dataset',
    '-d',
    'dataset_names',
    multiple=True,
    help='Dataset to act on, active or not (repeatable; default: every '
    'registered dataset -- `doctor` narrows this to active unless '
    '--include-inactive).',
)


def get_dataset(snowdb: SnowDb, name: str) -> Dataset:
    """Resolve a dataset by name, or raise a clean CLI error listing the options.

    Anything *registered* resolves -- the management/diagnostics surface,
    where activation is irrelevant (``stats`` is the one reader-surface
    caller, and it resolves through ``SnowDb.__getitem__`` directly instead,
    so its registered-but-inactive name gets that surface's own pointed
    "activate it" hint).
    """
    if name in snowdb.registered:
        return snowdb.registered[name]
    registered = ', '.join(sorted(snowdb.registered)) or '(none)'
    raise click.ClickException(
        f'No such dataset: {name!r}. Registered datasets: {registered}.',
    )


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
