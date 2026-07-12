"""The ``windows`` command group: Windows-only admin tooling.

Nests :mod:`snowtool.cli.iis` (IIS site install/remove) and adds
``add-to-path``, which puts this tool's own install directory on the
machine-wide ``PATH`` so every user on the box -- not just the one who ran
``uv tool install`` -- gets ``snowtool`` for free. ``uv tool install``
has no post-install hook to do this automatically, and ``uv tool
update-shell`` only ever touches the *current user's* PATH, so this has to
be a command an admin runs once, by hand, in an elevated shell.

That only works if snowtool itself was installed somewhere shared rather
than under the installing admin's own profile (``uv tool install``'s
default) -- otherwise every other user's PATH would point at a directory
under an account they may not have access to. ``add-to-path`` refuses to
proceed against a per-user install and prints how to reinstall with
``UV_TOOL_DIR``/``UV_TOOL_BIN_DIR``/``UV_PYTHON_INSTALL_DIR`` pointed at
shared locations instead (see ``CONTRIBUTING.md``). The last of those
covers the uv-managed interpreter the tool venv trampolines onto -- left
per-user, the venv works only for the installing account even from a
shared ``UV_TOOL_DIR``.
"""

from __future__ import annotations

import sys

from pathlib import Path

import click

from snowtool.cli import _path_env
from snowtool.cli._windows_common import require_admin, require_windows
from snowtool.cli.iis import iis


@click.group(hidden=sys.platform != 'win32')
def windows() -> None:
    """Windows-only admin commands (IIS deployment, all-users PATH setup)."""


windows.add_command(iis)


@windows.command('add-to-path')
def add_to_path() -> None:
    """Add this tool's install directory to the machine-wide PATH.

    Requires an elevated (Administrator) shell, since it writes the
    machine-scope ``Path`` registry value. Errors out (with reinstall
    instructions) if this install lives under a user profile rather than a
    shared location, since that would put a per-user path on the
    machine-wide PATH.
    """
    require_windows()
    require_admin()

    directory = Path(sys.executable).parent
    root = _path_env.users_root()
    if _path_env.is_user_specific_install(directory, root):
        raise click.ClickException(
            _path_env.user_specific_install_message(directory, root),
        )

    current = _path_env.read_system_path()
    if _path_env.contains_entry(current, directory):
        click.echo(f'{directory} is already on the system PATH.')
        return

    _path_env.write_system_path(_path_env.append_entry(current, directory))
    click.echo(f'Added {directory} to the system PATH for all users.')
    click.echo('Open a new shell for the change to take effect.')
