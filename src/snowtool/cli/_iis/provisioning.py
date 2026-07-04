"""Pure argv-building for the PowerShell IIS provisioning scripts.

:func:`run_powershell` is the one true I/O boundary here (an external Windows
process) -- callers build argv with the pure functions below, then pass it
through. ``runner`` defaults to :func:`subprocess.run` but is an injectable
seam so tests can capture the exact argv without a real ``powershell.exe``/IIS
host: the same category of exception CLAUDE.md carves out for monkeypatching a
true boundary (there: the network client; here: the OS process boundary).

Every value passed to PowerShell arrives as a bound ``-File`` script
parameter (see ``install_site.ps1``/``remove_site.ps1``), never interpolated
into a ``-Command`` string, so a site name or hostname can't inject
PowerShell.
"""

from __future__ import annotations

import importlib.resources
import subprocess

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

_INSTALL_SCRIPT = 'install_site.ps1'
_REMOVE_SCRIPT = 'remove_site.ps1'


def _script_path(name: str) -> Path:
    return Path(str(importlib.resources.files('snowtool.cli._iis').joinpath(name)))


def venv_root(python_exe: Path) -> Path:
    """The venv root containing ``python_exe`` (its ``Scripts``/``bin`` parent).

    Used to scope the ICACLS read+execute grant to the whole uv-tool venv
    (interpreter, standard library, installed packages), not just the
    executable itself.
    """
    return python_exe.parent.parent


def snowdb_root(snowdb_config: Path) -> Path:
    """The snowdb root directory for ``snowdb_config`` (file-or-directory).

    ``--config``/``SNOWTOOL_SNOWDB_CONFIG`` accepts either the root config
    file or its directory; the ICACLS grant always targets the directory, so
    a bare file path is resolved to its parent.
    """
    return snowdb_config if snowdb_config.is_dir() else snowdb_config.parent


def install_args(
    *,
    site_name: str,
    physical_path: Path,
    venv_path: Path,
    snowdb_path: Path,
    hostname: str,
    port: int,
    protocol: str,
    cert_thumbprint: str | None,
    recycle_time: str,
    access_log_dir: Path | None,
) -> list[str]:
    """The ``powershell`` argv that provisions (or updates) the app pool + site."""
    args = [
        'powershell',
        '-NoProfile',
        '-NonInteractive',
        '-ExecutionPolicy',
        'Bypass',
        '-File',
        str(_script_path(_INSTALL_SCRIPT)),
        '-SiteName',
        site_name,
        '-PhysicalPath',
        str(physical_path),
        '-VenvPath',
        str(venv_path),
        '-SnowdbPath',
        str(snowdb_path),
        '-Hostname',
        hostname,
        '-Port',
        str(port),
        '-Protocol',
        protocol,
        '-RecycleTime',
        recycle_time,
    ]
    if cert_thumbprint:
        args += ['-CertThumbprint', cert_thumbprint]
    if access_log_dir is not None:
        args += ['-AccessLogDir', str(access_log_dir)]
    return args


def remove_args(*, site_name: str) -> list[str]:
    """The ``powershell`` argv that tears down the app pool + site."""
    return [
        'powershell',
        '-NoProfile',
        '-NonInteractive',
        '-ExecutionPolicy',
        'Bypass',
        '-File',
        str(_script_path(_REMOVE_SCRIPT)),
        '-SiteName',
        site_name,
    ]


def run_powershell(
    args: list[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> subprocess.CompletedProcess[bytes]:
    """Run a provisioning script's argv, raising if it exits nonzero."""
    return runner(args, check=True)
