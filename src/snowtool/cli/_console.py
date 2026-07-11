"""The CLI's two rich consoles: stdout for data, stderr for everything else.

Data (tables, json, csv) renders through :func:`out`; progress, status
messages, and summaries render through :func:`err` -- so piped stdout stays
machine-clean while a TTY still gets the full interface. The root ``cli``
group's ``--color``/``--quiet`` options call :func:`configure`, which rebinds
the pair; callers must therefore fetch a console at use time, never hold one
from import time. rich handles TTY detection and ``NO_COLOR`` itself.
"""

from __future__ import annotations

from rich.console import Console

_FORCE = {'auto': None, 'always': True, 'never': False}

_out = Console()
_err = Console(stderr=True)


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
    _out = Console(force_terminal=force)
    _err = Console(stderr=True, force_terminal=force, quiet=quiet)
