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

import sys

from pathlib import Path

import click

from snowtool.cli._iis.provisioning import (
    base_python_root,
    install_args,
    remove_args,
    run_powershell,
    snowdb_root,
    venv_root,
)
from snowtool.cli._iis.web_config import rasterio_data_env, render_web_config
from snowtool.cli._windows_common import require_windows


def _resolve_site_name(directory: Path, site_name: str | None) -> str:
    return site_name if site_name is not None else directory.name


@click.group()
def iis() -> None:
    """Install/remove the API as an IIS site (Windows only)."""


@iis.command('install')
@click.argument('directory', type=click.Path(path_type=Path))
@click.option('--hostname', required=True, help='Hostname to bind the site to.')
@click.option('--port', type=int, default=443, show_default=True, help='Port to bind.')
@click.option(
    '--protocol',
    type=click.Choice(['http', 'https']),
    default='https',
    show_default=True,
)
@click.option(
    '--config',
    '-C',
    'snowdb_config',
    required=True,
    envvar='SNOWTOOL_SNOWDB_CONFIG',
    type=click.Path(path_type=Path),
    help='Snowdb config the hosted process reads (defaults to the '
    'SNOWTOOL_SNOWDB_CONFIG env var). Its directory is granted read+execute '
    "access for the site's app pool identity.",
)
@click.option(
    '--site-name',
    default=None,
    help='IIS site/app-pool name (defaults to the install directory name).',
)
@click.option(
    '--cert-thumbprint',
    default=None,
    help='SSL certificate thumbprint to bind (https only). '
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
    '--skip-site',
    is_flag=True,
    default=False,
    help='Only write web.config.',
)
@click.option(
    '--skip-config',
    is_flag=True,
    default=False,
    help='Only provision the site.',
)
def install(
    directory: Path,
    hostname: str,
    port: int,
    protocol: str,
    snowdb_config: Path,
    site_name: str | None,
    cert_thumbprint: str | None,
    recycle_time: str,
    access_log_dir: Path | None,
    skip_site: bool,
    skip_config: bool,
) -> None:
    """Install (or update) DIRECTORY as an IIS site fronting the API.

    DIRECTORY is the site's physical path -- where web.config and IIS logs
    live -- independent of where snowtool itself is installed. Its parent
    must already exist; DIRECTORY and its log/ subdirectory are created if
    missing. Idempotent: re-running against an existing site updates it in
    place.
    """
    require_windows()
    site_name = _resolve_site_name(directory, site_name)

    if not directory.parent.is_dir():
        raise click.ClickException(f'{directory.parent} does not exist.')
    directory.mkdir(exist_ok=True)
    (directory / 'log').mkdir(exist_ok=True)

    if not skip_config:
        web_config = directory / 'web.config'
        web_config.write_text(
            render_web_config(
                Path(sys.executable),
                snowdb_config,
                data_env=rasterio_data_env(),
            ),
        )
        click.echo(f'Wrote {web_config}')

    if protocol == 'https' and not cert_thumbprint:
        click.echo(
            'No --cert-thumbprint given; after install, bind the SSL certificate '
            f'manually: IIS Manager > Sites > {site_name} > Edit Bindings.',
        )

    if skip_site:
        return

    click.echo(f'Provisioning IIS site {site_name!r}...')
    run_powershell(
        install_args(
            site_name=site_name,
            physical_path=directory,
            venv_path=venv_root(Path(sys.executable)),
            base_python_path=base_python_root(sys.prefix, sys.base_prefix),
            snowdb_path=snowdb_root(snowdb_config),
            hostname=hostname,
            port=port,
            protocol=protocol,
            cert_thumbprint=cert_thumbprint,
            recycle_time=recycle_time,
            access_log_dir=access_log_dir,
        ),
    )
    click.echo('Done.')


@iis.command('remove')
@click.argument('directory', type=click.Path(path_type=Path))
@click.option(
    '--config',
    '-C',
    'snowdb_config',
    required=True,
    envvar='SNOWTOOL_SNOWDB_CONFIG',
    type=click.Path(path_type=Path),
    help='Snowdb config the site was installed with (defaults to the '
    "SNOWTOOL_SNOWDB_CONFIG env var). Its directory's app-pool permission "
    'grant is removed.',
)
@click.option(
    '--site-name',
    default=None,
    help='IIS site/app-pool name (defaults to the install directory name).',
)
def remove(directory: Path, snowdb_config: Path, site_name: str | None) -> None:
    """Remove the IIS site + app pool installed at DIRECTORY.

    Also strips the app-pool account's permission grants from the venv, the
    snowdb directory, and DIRECTORY. Leaves DIRECTORY itself in place (it
    may hold logs) but deletes its web.config.
    """
    require_windows()
    site_name = _resolve_site_name(directory, site_name)

    click.echo(f'Removing IIS site {site_name!r}...')
    run_powershell(
        remove_args(
            site_name=site_name,
            venv_path=venv_root(Path(sys.executable)),
            base_python_path=base_python_root(sys.prefix, sys.base_prefix),
            snowdb_path=snowdb_root(snowdb_config),
            physical_path=directory,
        ),
    )

    web_config = directory / 'web.config'
    if web_config.exists():
        web_config.unlink()
        click.echo(f'Removed {web_config}')
    click.echo('Done.')
