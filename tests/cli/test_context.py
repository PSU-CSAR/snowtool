"""CLI context wiring: a single, lazily-built SnowDb per invocation."""

from pathlib import Path

import click

from click.testing import CliRunner

from snowtool.cli import cli
from snowtool.cli._context import CliContext, pass_snowdb


def test_context_builds_snowdb_lazily_and_once(tmp_path):
    ctx = CliContext(root=tmp_path)
    # Nothing is built until snowdb is read.
    assert ctx._snowdb is None

    first = ctx.snowdb
    second = ctx.snowdb

    assert first is second
    assert first.path == tmp_path


def test_context_falls_back_to_settings(tmp_path, monkeypatch):
    # With no --root, the snowdb_path setting supplies the root.
    monkeypatch.setenv('SNOWTOOL_SNOWDB_PATH', str(tmp_path))
    ctx = CliContext(root=None)

    assert ctx.snowdb.path == tmp_path


def test_version_does_not_build_a_snowdb(monkeypatch):
    # version must work with no snowdb_path configured: if it tried to build a
    # SnowDb, Settings() would raise for the missing setting and exit nonzero.
    monkeypatch.delenv('SNOWTOOL_SNOWDB_PATH', raising=False)

    result = CliRunner().invoke(cli, ['version'])

    assert result.exit_code == 0
    assert result.output.strip()


def test_migration_help_does_not_build_a_snowdb(monkeypatch):
    monkeypatch.delenv('SNOWTOOL_SNOWDB_PATH', raising=False)

    result = CliRunner().invoke(cli, ['migration', '--help'])

    assert result.exit_code == 0


def test_root_help_does_not_build_a_snowdb(monkeypatch):
    monkeypatch.delenv('SNOWTOOL_SNOWDB_PATH', raising=False)

    result = CliRunner().invoke(cli, ['--help'])

    assert result.exit_code == 0
    assert '--root' in result.output


def _app() -> click.Group:
    """A throwaway CLI exercising the real --root wiring + pass_snowdb."""

    @click.group()
    @click.option('--root', type=click.Path(path_type=Path), default=None)
    @click.pass_context
    def app(ctx: click.Context, root: Path | None) -> None:
        ctx.obj = CliContext(root=root)

    @app.command()
    @pass_snowdb
    def show(snowdb) -> None:
        click.echo(str(snowdb.path))

    return app


def test_pass_snowdb_injects_root_db(tmp_path):
    result = CliRunner().invoke(_app(), ['--root', str(tmp_path), 'show'])

    assert result.exit_code == 0
    assert result.output.strip() == str(tmp_path)


def test_pass_snowdb_uses_settings_without_root(tmp_path, monkeypatch):
    monkeypatch.setenv('SNOWTOOL_SNOWDB_PATH', str(tmp_path))

    result = CliRunner().invoke(_app(), ['show'])

    assert result.exit_code == 0
    assert result.output.strip() == str(tmp_path)
