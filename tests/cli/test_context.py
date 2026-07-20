"""CLI context wiring: a single, lazily-built SnowDb per invocation."""

import json

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
    assert first.root == tmp_path


def test_config_option_resolves_from_env_var(tmp_path, monkeypatch):
    # With no --config flag, the option's SNOWTOOL_SNOWDB_CONFIG envvar supplies the
    # root -- resolved by click on the command, not by CliContext reading Settings.
    SnowDbManager.initialize(tmp_path)
    monkeypatch.setenv('SNOWTOOL_SNOWDB_CONFIG', str(tmp_path))

    result = CliRunner().invoke(cli, ['dataset', 'list', '--format', 'json'])

    assert result.exit_code == 0
    assert result.output.strip() == '[]'


def test_snowdb_without_config_is_a_clean_error(monkeypatch):
    # No flag, no env var -> a clean ClickException, not a traceback.
    monkeypatch.delenv('SNOWTOOL_SNOWDB_CONFIG', raising=False)

    result = CliRunner().invoke(cli, ['dataset', 'list'])

    assert result.exit_code != 0
    assert 'No snowdb configured' in result.output


def test_no_ambient_env_var_leaves_injected_config_untouched(cli_obj):
    # The suite's autouse `_no_ambient_snowdb_config` fixture (tests/cli/conftest.py)
    # strips SNOWTOOL_SNOWDB_CONFIG for every CLI test, so an injected CliContext
    # (cli_obj serves the synthetic 'test' dataset) is never clobbered by a
    # maintainer's exported env var -- the case the old `_apply_config`
    # env-vs-flag branch used to guard explicitly, now covered by never letting
    # the ambient env var reach the test in the first place.
    result = CliRunner().invoke(
        cli,
        ['dataset', 'list', '--format', 'json'],
        obj=cli_obj,
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == [{'dataset': 'test', 'active': True}]


def test_explicit_config_flag_still_overrides_injected_config(cli_obj, tmp_path):
    # An explicit --config flag always wins over an injected CliContext: a
    # *different* initialized (empty) root on the command line replaces cli_obj's.
    other = tmp_path / 'other'
    SnowDbManager.initialize(other)

    result = CliRunner().invoke(
        cli,
        ['dataset', 'list', '--format', 'json', '--config', str(other)],
        obj=cli_obj,
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == []


def test_malformed_config_is_a_clean_cli_error(tmp_path):
    # A file exists but isn't a valid root config: a clean CLI error, not a traceback.
    from snowtool.snowdb.config import CONFIG_FILENAME

    (tmp_path / CONFIG_FILENAME).write_text('{ not valid json')

    result = CliRunner().invoke(cli, ['dataset', 'list', '--config', str(tmp_path)])

    assert result.exit_code != 0
    assert 'not a readable snowdb root config' in result.output
    assert 'Traceback' not in result.output


@pytest.mark.parametrize(
    ('args', 'expected_in_output'),
    [
        # --version must work with no snowdb_config configured: if it tried to build
        # a SnowDb, Settings() would raise for the missing setting and exit nonzero.
        (['--version'], None),
        (['--help'], 'Commands'),
        # --config is a per-command option now, shown on a command's help (and its
        # help short-circuits before the callback, so no SnowDb is built).
        (['dataset', 'list', '--help'], '--config'),
        # serve surfaces the same --config, not gazebo's --snowtool-snowdb-config.
        (['api', 'serve', '--help'], '--config'),
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
        click.echo(str(snowdb.root))

    @app.command('show-mgr')
    @pass_manager
    def show_mgr(manager) -> None:
        click.echo(str(manager.db.root))

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
