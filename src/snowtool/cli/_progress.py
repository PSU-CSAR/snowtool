"""Bind the domain progress seam to rich for the CLI.

The domain reports progress through
:class:`~snowtool.snowdb.progress.ProgressReporter` (a no-op by default); the
CLI passes :class:`RichProgress` so heavy operations (terrain reprojection, the
NLCD download, rasterizing many basins) render a live bar -- or a spinner when
the total is unknown. Rendering goes to the **stderr** console so
machine-readable stdout stays clean; on a non-TTY (piped, CI, tests) the label
is announced once and advancement is silent, and ``--quiet`` suppresses even
that (the err console is quiet).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from snowtool.cli import _console
from snowtool.snowdb.progress import NULL_PROGRESS

if TYPE_CHECKING:
    from collections.abc import Iterator

    from rich.console import Console

    from snowtool.snowdb.progress import ProgressTask


class _RichTask:
    def __init__(self: _RichTask, progress: Progress, task_id: TaskID) -> None:
        self._progress = progress
        self._task_id = task_id

    def advance(self: _RichTask, n: int = 1) -> None:
        self._progress.advance(self._task_id, n)


class RichProgress:
    """A :class:`~snowtool.snowdb.progress.ProgressReporter` backed by rich.

    ``prefix`` is prepended to every task label, so the CLI can name the context
    (e.g. ``'snodas ingest: '``) that the generic domain labels lack. ``console``
    defaults to the CLI's stderr console, fetched at track time so the root
    ``--color``/``--quiet`` configuration applies.
    """

    def __init__(
        self: RichProgress,
        prefix: str = '',
        console: Console | None = None,
    ) -> None:
        self._prefix = prefix
        self._console = console

    @contextmanager
    def track(
        self: RichProgress,
        label: str,
        *,
        total: int | None = None,
    ) -> Iterator[ProgressTask]:
        console = self._console if self._console is not None else _console.err()
        label = f'{self._prefix}{label}'
        if not console.is_terminal:
            # Piped/CI: announce the step once so a redirected log still shows
            # it, then advance silently through the domain no-op task.
            console.print(f'{label}...', highlight=False)
            with NULL_PROGRESS.track(label, total=total) as task:
                yield task
            return
        progress = Progress(
            SpinnerColumn(),
            TextColumn('[progress.description]{task.description}'),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
        with progress:
            yield _RichTask(progress, progress.add_task(label, total=total))
