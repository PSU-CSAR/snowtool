"""Orchestration + pure argv-building for the PowerShell IIS provisioning.

:func:`install_site`/:func:`remove_site` own the end-to-end install/remove
sequencing (directory creation, ``web.config`` render+write, ``--only`` step
gating, cert messaging, teardown-then-unlink ordering) so the
:mod:`snowtool.cli.iis` click callbacks stay the thin guard+call+echo shape
every other command in this CLI has. They take their two I/O seams --
``echo`` (user-facing output) and ``runner`` (the PowerShell process) -- as
injectable parameters, defaulting to :func:`print`/:func:`subprocess.run`, so
the whole sequence is unit-testable off Windows without a real
``powershell.exe``/IIS host or click's runner.

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
import sys

from pathlib import Path
from typing import TYPE_CHECKING

import click

from snowtool.cli._iis.web_config import rasterio_data_env, render_web_config

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


def base_python_root(prefix: str, base_prefix: str) -> Path | None:
    """The base interpreter install backing a venv, or ``None`` outside one.

    On Windows a uv venv's ``python.exe`` is a trampoline onto the base
    interpreter recorded in ``pyvenv.cfg`` -- which for uv-managed pythons
    defaults to the installing user's profile -- so the app-pool account
    needs read+execute there too, not just on the venv. Call with
    ``sys.prefix``/``sys.base_prefix``.
    """
    return Path(base_prefix) if base_prefix != prefix else None


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
    base_python_path: Path | None,
    snowdb_path: Path,
    hostname: str,
    port: int,
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
        'https',
        '-RecycleTime',
        recycle_time,
    ]
    if base_python_path is not None:
        args += ['-BasePythonPath', str(base_python_path)]
    if cert_thumbprint:
        args += ['-CertThumbprint', cert_thumbprint]
    if access_log_dir is not None:
        args += ['-AccessLogDir', str(access_log_dir)]
    return args


def remove_args(
    *,
    site_name: str,
    venv_path: Path,
    base_python_path: Path | None,
    snowdb_path: Path,
    physical_path: Path,
) -> list[str]:
    """The ``powershell`` argv that tears down the app pool + site.

    The paths are where ``install_args``'s script granted the app-pool
    account permissions; the remove script strips those grants again.
    """
    args = [
        'powershell',
        '-NoProfile',
        '-NonInteractive',
        '-ExecutionPolicy',
        'Bypass',
        '-File',
        str(_script_path(_REMOVE_SCRIPT)),
        '-SiteName',
        site_name,
        '-VenvPath',
        str(venv_path),
        '-SnowdbPath',
        str(snowdb_path),
        '-PhysicalPath',
        str(physical_path),
    ]
    if base_python_path is not None:
        args += ['-BasePythonPath', str(base_python_path)]
    return args


def run_powershell(
    args: list[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> subprocess.CompletedProcess[bytes]:
    """Run a provisioning script's argv, raising if it exits nonzero."""
    return runner(args, check=True)


def _resolve_site_name(site_name: str | None, directory: Path) -> str:
    """The IIS site/app-pool name, defaulting to the install directory name."""
    return site_name if site_name is not None else directory.name


def install_site(
    *,
    directory: Path,
    hostname: str,
    port: int,
    snowdb_config: Path,
    site_name: str | None,
    cert_thumbprint: str | None,
    recycle_time: str,
    access_log_dir: Path | None,
    only: str | None,
    echo: Callable[[str], None] = click.echo,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> None:
    """Install (or update) ``directory`` as an IIS site fronting the API.

    Owns the whole install sequence: validate the parent, create the site +
    ``log/`` directories, render+write ``web.config`` (unless ``--only site``),
    then provision the IIS site via PowerShell (unless ``--only config``). The
    two ordered steps are gated by ``only``; the ``web.config`` write always
    precedes provisioning. ``echo``/``runner`` are injectable I/O seams so the
    sequence is testable off Windows.
    """
    site_name = _resolve_site_name(site_name, directory)

    if not directory.parent.is_dir():
        raise click.ClickException(f'{directory.parent} does not exist.')
    directory.mkdir(exist_ok=True)
    (directory / 'log').mkdir(exist_ok=True)

    if only != 'site':
        web_config = directory / 'web.config'
        web_config.write_text(
            render_web_config(
                Path(sys.executable),
                snowdb_config,
                data_env=rasterio_data_env(),
            ),
        )
        echo(f'Wrote {web_config}')

    if only != 'config':
        if not cert_thumbprint:
            echo(
                'No --cert-thumbprint given; after install, bind the SSL '
                f'certificate manually: IIS Manager > Sites > {site_name} > '
                'Edit Bindings.',
            )

        echo(f'Provisioning IIS site {site_name!r}...')
        run_powershell(
            install_args(
                site_name=site_name,
                physical_path=directory,
                venv_path=venv_root(Path(sys.executable)),
                base_python_path=base_python_root(sys.prefix, sys.base_prefix),
                snowdb_path=snowdb_root(snowdb_config),
                hostname=hostname,
                port=port,
                cert_thumbprint=cert_thumbprint,
                recycle_time=recycle_time,
                access_log_dir=access_log_dir,
            ),
            runner=runner,
        )
        echo('Done.')


def remove_site(
    *,
    directory: Path,
    snowdb_config: Path,
    site_name: str | None,
    echo: Callable[[str], None] = click.echo,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> None:
    """Remove the IIS site + app pool installed at ``directory``.

    Owns the whole remove sequence: tear down the site + strip the app-pool
    permission grants via PowerShell *first*, then unlink ``web.config`` (the
    teardown-before-unlink ordering leaves nothing half-removed if the script
    fails). ``echo``/``runner`` are injectable I/O seams so the sequence is
    testable off Windows.
    """
    site_name = _resolve_site_name(site_name, directory)

    echo(f'Removing IIS site {site_name!r}...')
    run_powershell(
        remove_args(
            site_name=site_name,
            venv_path=venv_root(Path(sys.executable)),
            base_python_path=base_python_root(sys.prefix, sys.base_prefix),
            snowdb_path=snowdb_root(snowdb_config),
            physical_path=directory,
        ),
        runner=runner,
    )

    web_config = directory / 'web.config'
    if web_config.exists():
        web_config.unlink()
        echo(f'Removed {web_config}')
    echo('Done.')
