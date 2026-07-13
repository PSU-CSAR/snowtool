"""Pure PATH-string logic for ``snowtool windows add-to-path``.

The registry read/write at the bottom is the one true I/O boundary here (an
external Windows subsystem, same category as ``run_powershell`` in
``snowtool.cli._iis.provisioning``) -- ``winreg``/``ctypes.windll`` are
imported lazily inside those two functions so the module stays importable
(and its pure logic testable) on any platform. Everything above them is pure
string/path manipulation.
"""

from __future__ import annotations

import importlib
import os

from pathlib import PureWindowsPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_ENV_KEY = r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment'


def users_root(environ: Mapping[str, str] | None = None) -> PureWindowsPath:
    """The parent of user profile directories (``C:\\Users``, by convention).

    Derived from ``%USERPROFILE%`` rather than hardcoded; ``environ`` is
    injectable so tests don't depend on a real Windows environment.
    """
    environ = os.environ if environ is None else environ
    return PureWindowsPath(environ['USERPROFILE']).parent


def is_user_specific_install(directory: Path, users_root: PureWindowsPath) -> bool:
    """Whether ``directory`` sits inside a per-user profile.

    True for the ``uv tool install`` default (e.g.
    ``%USERPROFILE%\\.local\\bin``); false once an admin has pointed
    ``UV_TOOL_BIN_DIR`` at a shared location such as ``C:\\ProgramData\\...``.
    """
    try:
        PureWindowsPath(directory).relative_to(users_root)
    except ValueError:
        return False
    return True


def user_specific_install_message(directory: Path, users_root: PureWindowsPath) -> str:
    """The error text guiding a per-user install to a shared reinstall."""
    return (
        f'{directory} is a per-user install (under {users_root}), so adding '
        'it to the machine-wide PATH would only work for the account that '
        'installed it.\n\n'
        'Reinstall snowtool as a shared, all-users tool first, in an '
        'elevated shell:\n\n'
        '  setx /M UV_TOOL_DIR C:\\ProgramData\\uv\\tools\n'
        '  setx /M UV_TOOL_BIN_DIR C:\\ProgramData\\uv\\bin\n'
        '  setx /M UV_PYTHON_INSTALL_DIR C:\\ProgramData\\uv\\python\n'
        '  setx /M UV_LINK_MODE copy\n'
        '  <start a new elevated shell so the env vars take effect>\n'
        '  uv tool install snowtool\n\n'
        'UV_PYTHON_INSTALL_DIR keeps the Python interpreter backing the '
        "tool venv out of the installing user's profile, where other "
        'accounts (like an IIS app pool) cannot read it. UV_LINK_MODE=copy '
        'keeps installed files from being hardlinks into the per-user uv '
        'cache, which stay locked to the installing user no matter where '
        'they appear to live.\n\n'
        'Then re-run `snowtool windows add-to-path`.'
    )


def _normalize(entry: str) -> str:
    return str(PureWindowsPath(entry.rstrip('\\'))).casefold()


def path_entries(value: str) -> list[str]:
    """Split a Windows ``Path``-variable value into its non-empty entries."""
    return [entry for entry in value.split(';') if entry]


def contains_entry(value: str, directory: Path) -> bool:
    """Whether ``directory`` is already among ``value``'s entries."""
    target = _normalize(str(directory))
    return any(_normalize(entry) == target for entry in path_entries(value))


def append_entry(value: str, directory: Path) -> str:
    """``value`` with ``directory`` appended, preserving existing entries."""
    return f'{value.rstrip(";")};{directory}' if value else str(directory)


def read_system_path() -> str:
    """The machine-wide ``Path`` value from the registry."""
    # typeshed only exposes winreg's attributes when checking under
    # win32, so this module (developed cross-platform) treats it as Any.
    winreg: Any = importlib.import_module('winreg')

    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _ENV_KEY, 0, winreg.KEY_READ)
    try:
        value, _ = winreg.QueryValueEx(key, 'Path')
    finally:
        key.Close()
    return str(value)


def write_system_path(value: str) -> None:
    """Write the machine-wide ``Path`` value and notify running processes."""
    winreg: Any = importlib.import_module('winreg')

    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _ENV_KEY, 0, winreg.KEY_SET_VALUE)
    try:
        winreg.SetValueEx(key, 'Path', 0, winreg.REG_EXPAND_SZ, value)
    finally:
        key.Close()
    _broadcast_environment_change()


def _broadcast_environment_change() -> None:
    """Tell running processes (e.g. Explorer) that Environment changed.

    Without this, only shells started after the registry write see the new
    PATH; broadcasting lets Explorer refresh its own environment block so
    shells launched from it afterward inherit the change too.
    """
    import ctypes

    # ``windll`` is typeshed-gated to win32, same rationale as the winreg
    # import above.
    windll: Any = ctypes.windll  # type: ignore[attr-defined]

    result = ctypes.c_long()
    windll.user32.SendMessageTimeoutW(
        0xFFFF,  # HWND_BROADCAST
        0x1A,  # WM_SETTINGCHANGE
        0,
        'Environment',
        0x0002,  # SMTO_ABORTIFHUNG
        5000,
        ctypes.byref(result),
    )
