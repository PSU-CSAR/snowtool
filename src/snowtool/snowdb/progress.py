"""A tiny progress-reporting seam for long domain operations.

Some operations take minutes -- terrain reprojects a DEM block by block, the Annual
NLCD source downloads ~1.5 GB -- so they report progress. But the domain must not
depend on the CLI: instead it takes a :class:`ProgressReporter` (default
:data:`NULL_PROGRESS`, a no-op) and, for each long task, opens a
:meth:`~ProgressReporter.track` context with a label and an optional total,
advancing it as units of work complete. The CLI binds this to ``click.progressbar``
(see ``snowtool.cli._progress``); the API and tests use the null reporter and pay
nothing.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator
    from contextlib import AbstractContextManager


class ProgressTask(Protocol):
    """A single in-flight task: advance it as units of work complete."""

    def advance(self, n: int = 1) -> None: ...


class ProgressReporter(Protocol):
    """Opens a tracked task for a long operation.

    ``total`` is the expected unit count when known (block count, byte length), or
    ``None`` for an indeterminate task. The returned context manager yields a
    :class:`ProgressTask`; the task is finished when the context exits.
    """

    def track(
        self: ProgressReporter,
        label: str,
        *,
        total: int | None = None,
    ) -> AbstractContextManager[ProgressTask]: ...


class _NullTask:
    def advance(self: _NullTask, n: int = 1) -> None:
        pass


class NullProgress:
    """The default no-op reporter, so domain code can report unconditionally."""

    @contextmanager
    def track(
        self: NullProgress,
        label: str,
        *,
        total: int | None = None,
    ) -> Iterator[ProgressTask]:
        yield _NullTask()


NULL_PROGRESS: ProgressReporter = NullProgress()
