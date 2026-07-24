"""The ``iis`` command group: install/remove the API as a Windows IIS site.

Deploys ``snowtool api serve`` behind IIS's httpPlatformHandler, which shells
out to a child process and reverse-proxies to it. ``install`` renders a
``web.config`` (:mod:`snowtool.cli._iis.web_config`) and idempotently
provisions (or updates) the app pool + site via a bundled PowerShell script
(:mod:`snowtool.cli._iis.provisioning`); ``remove`` tears both down. Neither
command opens a snowdb *in this process*, so -- like :mod:`snowtool.cli.api`
-- this group takes no :class:`~snowtool.cli._context.CliContext`/
``pass_snowdb``; its own ``--config``/``-C`` carries a path through into
``web.config`` as the ``SNOWTOOL_SNOWDB_CONFIG`` environment variable the
hosted process reads, and its resolved directory
(:func:`~snowtool.cli._iis.provisioning.snowdb_root`) is also granted
read+execute for the site's app pool identity -- the hosted API needs to
read it at request time, same as the venv it runs out of.

Nested under :mod:`snowtool.cli.windows` (``snowtool windows iis ...``),
alongside ``add-to-path``. Windows is checked at command-body time, not
import time, so ``snowtool windows iis --help`` still works on any platform.
"""

from __future__ import annotations

from pathlib import Path

import click

from snowtool.cli._iis.provisioning import install_site, remove_site
from snowtool.cli._windows_common import require_windows

_config_option = click.option(
    '--config',
    '-C',
    'snowdb_config',
    required=True,
    envvar='SNOWTOOL_SNOWDB_CONFIG',
    type=click.Path(path_type=Path),
    help='Snowdb config the site is installed with (defaults to the '
    "SNOWTOOL_SNOWDB_CONFIG env var). Its directory's app-pool permission "
    'grant is set (install) or removed (remove).',
)
_site_name_option = click.option(
    '--site-name',
    default=None,
    help='IIS site/app-pool name (defaults to the install directory name).',
)


@click.group()
def iis() -> None:
    """Install/remove the API as an IIS site (Windows only)."""


@iis.command('install')
@click.argument('directory', type=click.Path(path_type=Path))
@click.option('--hostname', required=True, help='Hostname to bind the site to.')
@click.option('--port', type=int, default=443, show_default=True, help='Port to bind.')
@_config_option
@_site_name_option
@click.option(
    '--cert-thumbprint',
    default=None,
    help='SSL certificate thumbprint to bind. '
    'Omit to bind the certificate manually afterward.',
)
@click.option(
    '--recycle-time',
    default='03:00:00',
    show_default=True,
    help='Fixed daily time (HH:MM:SS) the app pool periodically recycles at.',
)
@click.option(
    '--access-log-dir',
    'access_log_dir',
    default=None,
    type=click.Path(path_type=Path),
    help="IIS site access-log directory (defaults to IIS's own default).",
)
@click.option(
    '--only',
    'only',
    type=click.Choice(['config', 'site']),
    default=None,
    help='Run only one step instead of both: "config" writes web.config '
    'only; "site" provisions the IIS site only. Default: run both.',
)
def install(
    directory: Path,
    hostname: str,
    port: int,
    snowdb_config: Path,
    site_name: str | None,
    cert_thumbprint: str | None,
    recycle_time: str,
    access_log_dir: Path | None,
    only: str | None,
) -> None:
    """Install (or update) DIRECTORY as an IIS site fronting the API.

    DIRECTORY is the site's physical path -- where web.config and IIS logs
    live -- independent of where snowtool itself is installed. Its parent
    must already exist; DIRECTORY and its log/ subdirectory are created if
    missing. Idempotent: re-running against an existing site updates it in
    place.
    """
    require_windows()
    install_site(
        directory=directory,
        hostname=hostname,
        port=port,
        snowdb_config=snowdb_config,
        site_name=site_name,
        cert_thumbprint=cert_thumbprint,
        recycle_time=recycle_time,
        access_log_dir=access_log_dir,
        only=only,
    )


@iis.command('remove')
@click.argument('directory', type=click.Path(path_type=Path))
@_config_option
@_site_name_option
def remove(directory: Path, snowdb_config: Path, site_name: str | None) -> None:
    """Remove the IIS site + app pool installed at DIRECTORY.

    Also strips the app-pool account's permission grants from the venv, the
    snowdb directory, and DIRECTORY. Leaves DIRECTORY itself in place (it
    may hold logs) but deletes its web.config.
    """
    require_windows()
    remove_site(
        directory=directory,
        snowdb_config=snowdb_config,
        site_name=site_name,
    )
