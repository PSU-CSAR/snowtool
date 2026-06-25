"""CLI context wiring: a single, lazily-built SnowDb per invocation."""

from pathlib import Path

import click
import pytest

from click.testing import CliRunner

from snowtool.cli import cli
from snowtool.cli._context import CliContext, pass_manager, pass_snowdb
from snowtool.snowdb.manager import SnowDbManager


def test_context_builds_snowdb_lazily_and_once(tmp_path):
    SnowDbManager.initialize(tmp_path)  # open() requires a root config
    ctx = CliContext(config=tmp_path)
    # Nothing is built until snowdb is read.
    assert ctx._snowdb is None

    first = ctx.snowdb
    second = ctx.snowdb

    assert first is second
    assert first.path == tmp_path


def test_context_falls_back_to_settings(tmp_path, monkeypatch):
    # With no --config, the snowdb_config setting supplies the root.
    SnowDbManager.initialize(tmp_path)
    monkeypatch.setenv('SNOWTOOL_SNOWDB_CONFIG', str(tmp_path))
    ctx = CliContext(config=None)

    assert ctx.snowdb.path == tmp_path


@pytest.mark.parametrize(
    ('args', 'expected_in_output'),
    [
        # version must work with no snowdb_config configured: if it tried to build a
        # SnowDb, Settings() would raise for the missing setting and exit nonzero.
        (['version'], None),
        (['migration', '--help'], None),
        (['--help'], '--config'),
    ],
)
def test_command_does_not_build_a_snowdb(monkeypatch, args, expected_in_output):
    monkeypatch.delenv('SNOWTOOL_SNOWDB_CONFIG', raising=False)

    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 0
    if expected_in_output is not None:
        assert expected_in_output in result.output


def _app() -> click.Group:
    """A throwaway CLI exercising the real --root wiring + pass_snowdb."""

    @click.group()
    @click.option('--root', type=click.Path(path_type=Path), default=None)
    @click.pass_context
    def app(ctx: click.Context, root: Path | None) -> None:
        ctx.obj = CliContext(config=root)

    @app.command()
    @pass_snowdb
    def show(snowdb) -> None:
        click.echo(str(snowdb.path))

    @app.command('show-mgr')
    @pass_manager
    def show_mgr(manager) -> None:
        click.echo(str(manager.db.path))

    return app


@pytest.mark.parametrize('command', ['show', 'show-mgr'])
def test_pass_decorators_inject_the_root_db(tmp_path, command):
    # pass_snowdb (show) and pass_manager (show-mgr) both resolve the same lazily
    # opened SnowDb from --root and print its path.
    SnowDbManager.initialize(tmp_path)
    result = CliRunner().invoke(_app(), ['--root', str(tmp_path), command])

    assert result.exit_code == 0
    assert result.output.strip() == str(tmp_path)


def test_manager_wraps_the_lazy_snowdb(tmp_path):
    SnowDbManager.initialize(tmp_path)
    ctx = CliContext(config=tmp_path)

    assert ctx.manager.db is ctx.snowdb


def test_pass_snowdb_uses_settings_without_root(tmp_path, monkeypatch):
    SnowDbManager.initialize(tmp_path)
    monkeypatch.setenv('SNOWTOOL_SNOWDB_CONFIG', str(tmp_path))

    result = CliRunner().invoke(_app(), ['show'])

    assert result.exit_code == 0
    assert result.output.strip() == str(tmp_path)
