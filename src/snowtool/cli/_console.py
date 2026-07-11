"""The CLI's two rich consoles: stdout for data, stderr for everything else.

Data (tables, json, csv) renders through :func:`out`; progress, status
messages, and summaries render through :func:`err` -- so piped stdout stays
machine-clean while a TTY still gets the full interface. The root ``cli``
group's ``--color``/``--quiet`` options call :func:`configure`, which rebinds
the pair; callers must therefore fetch a console at use time, never hold one
from import time. rich handles TTY detection and ``NO_COLOR`` itself. A
non-terminal console is rebuilt with a huge fixed width, since piped output
must never wrap mid-word just because rich's non-TTY default is 80 columns.
"""

from __future__ import annotations

from rich.console import Console

_FORCE = {'auto': None, 'always': True, 'never': False}

# rich falls back to width=80 for a non-terminal stream; a wide table (e.g.
# a 10-column status report) then folds cell contents mid-word in exactly the
# piped/redirected output that must stay plain and parseable.
_NON_TERMINAL_WIDTH = 10_000


def _make(*, stderr: bool, force: bool | None, quiet: bool = False) -> Console:
    """Build a console, then widen it if it isn't attached to a terminal.

    A first console is needed to learn whether the underlying stream is a
    terminal; only a non-terminal one gets rebuilt at a fixed large width, so
    a real TTY keeps its detected width (and rich's normal wrapping there).
    """
    console = Console(stderr=stderr, force_terminal=force, quiet=quiet)
    if not console.is_terminal:
        console = Console(
            stderr=stderr,
            force_terminal=force,
            quiet=quiet,
            width=_NON_TERMINAL_WIDTH,
        )
    return console


_out = _make(stderr=False, force=None)
_err = _make(stderr=True, force=None)


def out() -> Console:
    """The data console (stdout)."""
    return _out


def err() -> Console:
    """The status console (stderr): progress, summaries, ok/problem lines."""
    return _err


def configure(*, color: str = 'auto', quiet: bool = False) -> None:
    """Rebind the console pair from the root ``--color``/``--quiet`` options.

    ``--quiet`` silences only the status side: data output is the command's
    result and is never suppressed.
    """
    global _out, _err
    force = _FORCE[color]
    _out = _make(stderr=False, force=force)
    _err = _make(stderr=True, force=force, quiet=quiet)
