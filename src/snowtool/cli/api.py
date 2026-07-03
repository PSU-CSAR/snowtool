"""The ``api`` command group: run the read-API server.

``serve`` is built by gazebo's :func:`serve_command` over the
``snowtool.api.app:get_app`` factory. That one helper documents *this app* --
one self-documenting flag per :class:`~snowtool.api.settings.Settings` field,
plus a ``--check`` preflight that validates settings + app import without
starting a server. uvicorn's own options (``--host``/``--port``/``--workers``/
``--reload``/...) are still accepted and forwarded verbatim to uvicorn, but no
longer clutter ``serve --help``; run ``serve --help-server`` to list them.

Config is surfaced under the CLI's usual ``--config``/``-C`` (as every other
command): a :class:`~gazebo.ext.cli.SettingsGroup` *renames* gazebo's generated
``--snowtool-snowdb-config`` to it, keeping the field's env var. serve runs the app
under uvicorn -- whose ``--reload``/``--workers`` subprocesses re-import and re-read
the environment -- and the renamed option self-propagates to ``SNOWTOOL_SNOWDB_CONFIG``
(the var the factory and those subprocesses read), so no :class:`CliContext` is
involved. Unlike the in-process commands, ``--config`` here stays *required* (the field
has no default): you cannot serve without a snowdb, so it fails fast and cleanly.

The command is built *lazily* (on first access) so the heavy
``gazebo.ext.uvicorn`` import (uvicorn, pydantic-settings) is paid only when the API
is actually served, never on a plain ``snowtool`` invocation.
"""

from __future__ import annotations

import click


class _ApiGroup(click.Group):
    """The ``api`` group, building its uvicorn-backed ``serve`` command on demand.

    Deferring the build keeps ``gazebo.ext.uvicorn`` (and uvicorn) out of the import
    path of every other ``snowtool`` command -- only ``snowtool api ...`` pays it.
    ``list_commands``/``get_command`` are the click seam click itself calls to
    render ``api --help`` and to resolve ``api serve``.
    """

    def list_commands(self, ctx: click.Context) -> list[str]:
        return ['serve']

    def get_command(self, ctx: click.Context, name: str) -> click.Command | None:
        if name == 'serve':
            return _build_serve_command()
        return None


def _build_serve_command() -> click.Command:
    from gazebo.ext.cli import SettingsGroup, default_log_config
    from gazebo.ext.uvicorn import serve_command

    from snowtool.api.settings import Settings

    # Rename gazebo's generated --snowtool-snowdb-config to the CLI's usual -C/--config,
    # so serve presents the same config flag as every other command. The rename keeps
    # the field's env var, so the option self-propagates to SNOWTOOL_SNOWDB_CONFIG (read
    # by the factory and any uvicorn reload/worker subprocesses) and --check still
    # validates it via Settings(). SettingsGroup raises if the renamed flag ever stops
    # matching a generated option, so a gazebo/field-name change fails loudly here.
    settings_group = SettingsGroup(
        Settings,
        rename={'--snowtool-snowdb-config': ['-C', '--config']},
    )
    return serve_command(
        'snowtool.api.app:get_app',
        factory=True,
        settings_group=settings_group,
        log_config=default_log_config(request_id=True),
    )


@click.group(cls=_ApiGroup)
def api() -> None:
    """Read-API server commands."""
