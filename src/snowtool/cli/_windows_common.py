"""Guards shared by every command under ``snowtool windows``.

Checked at command-body time, not import time, so ``snowtool windows
--help``/``snowtool windows iis --help`` still work on any platform.
"""

from __future__ import annotations

import sys

import click


def require_windows() -> None:
    if not sys.platform.startswith('win'):
        raise click.ClickException('snowtool windows commands must run on Windows.')


def require_admin() -> None:
    """Raise unless the current process is elevated (Administrator).

    Only called after :func:`require_windows` has already confirmed we're on
    Windows, so the ``ctypes.windll`` access below is safe.
    """
    import ctypes

    if not ctypes.windll.shell32.IsUserAnAdmin():  # type: ignore[attr-defined]
        raise click.ClickException(
            'snowtool windows commands must run in an elevated (Administrator) shell.',
        )
