"""Bind the domain progress seam to ``click.progressbar`` for the CLI.

The domain reports progress through
:class:`~snowtool.snowdb.progress.ProgressReporter` (a no-op by default); the CLI
passes :class:`ClickProgress` so heavy operations (terrain reprojection, the NLCD
download, rasterizing many basins) render a live bar. The bar is written to
**stderr** so machine-readable stdout stays clean, and click hides it when stderr
is not a TTY -- so piped and test runs stay silent while still advancing.
"""

from __future__ import annotations

import sys

from contextlib import contextmanager
from typing import TYPE_CHECKING

import click

from snowtool.snowdb.progress import NULL_PROGRESS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from snowtool.snowdb.progress import ProgressTask


class _BarTask:
    # Holds the bar's ``update`` bound method rather than the bar itself, so nothing
    # here depends on click's private ProgressBar type.
    def __init__(self: _BarTask, update: Callable[[int], None]) -> None:
        self._update = update

    def advance(self: _BarTask, n: int = 1) -> None:
        self._update(n)


class ClickProgress:
    """A :class:`~snowtool.snowdb.progress.ProgressReporter` backed by click.

    ``prefix`` is prepended to every task label, so the CLI can name the context
    (e.g. ``'snodas terrain: '``) that the generic domain labels lack.
    """

    def __init__(self: ClickProgress, prefix: str = '') -> None:
        self._prefix = prefix

    @contextmanager
    def track(
        self: ClickProgress,
        label: str,
        *,
        total: int | None = None,
    ) -> Iterator[ProgressTask]:
        label = f'{self._prefix}{label}'
        if total is None:
            # Indeterminate: announce the step, then hand back the domain no-op task.
            click.echo(f'{label}...', err=True)
            with NULL_PROGRESS.track(label) as task:
                yield task
            return
        # On a non-TTY click.progressbar renders no bar but still echoes the label
        # line once, so a backgrounded/CI run (`> log 2>&1`) shows the step without
        # any extra echo here.
        with click.progressbar(length=total, label=label, file=sys.stderr) as bar:
            yield _BarTask(bar.update)
