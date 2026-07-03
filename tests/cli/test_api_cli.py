"""The ``api serve`` command wiring (the gazebo ``serve_command`` glue).

The FastAPI app itself is covered by the router tests (which call
``get_app(settings=...)`` directly); these lock the *CLI* side -- that the lazy
``api`` group resolves ``serve``, and that ``serve --check`` validates settings and
imports the app factory (its preflight) without starting a server. Actually booting
uvicorn on a socket is left to a manual smoke -- too slow/flaky for the suite, and
the factory is already exercised directly.
"""

import os

from click.testing import CliRunner

from snowtool.cli import cli


def test_api_help_lists_serve():
    # Exercises the lazy _ApiGroup: --help must build + list the serve command.
    result = CliRunner().invoke(cli, ['api', '--help'])

    assert result.exit_code == 0
    assert 'serve' in result.output


def test_serve_help_uses_cli_config_flag(monkeypatch, tmp_path):
    # serve presents the CLI's --config, and hides gazebo's --snowtool-snowdb-config.
    monkeypatch.setenv('SNOWTOOL_SNOWDB_CONFIG', str(tmp_path))

    result = CliRunner().invoke(cli, ['api', 'serve', '--help'])

    assert result.exit_code == 0
    assert '--config' in result.output
    assert '--snowtool-snowdb-config' not in result.output


def test_serve_check_ok_with_config_flag(monkeypatch, tmp_path):
    # The renamed --config (gazebo's --snowtool-snowdb-config -> -C/--config) self-
    # propagates to the env var the factory/workers read. Register the var with
    # monkeypatch so its teardown restores the environment even though the option
    # writes it in-process.
    monkeypatch.setenv('SNOWTOOL_SNOWDB_CONFIG', 'placeholder')

    result = CliRunner().invoke(
        cli,
        ['api', 'serve', '--check', '--config', str(tmp_path)],
    )

    assert result.exit_code == 0
    assert 'OK' in result.output
    assert os.environ['SNOWTOOL_SNOWDB_CONFIG'] == str(tmp_path)


def test_serve_check_ok_with_config(monkeypatch, tmp_path):
    # --check runs Settings() + imports the app factory, then exits 0. Set the
    # config via the env var (not the flag) so the option's env write does not leak
    # past monkeypatch's teardown.
    monkeypatch.setenv('SNOWTOOL_SNOWDB_CONFIG', str(tmp_path))

    result = CliRunner().invoke(cli, ['api', 'serve', '--check'])

    assert result.exit_code == 0
    assert 'OK' in result.output


def test_serve_check_fails_without_config(monkeypatch):
    # --config is required (snowdb_config has no default), satisfied by the flag or the
    # env var. With neither set, click errors on the missing option before --check even
    # runs -- a clean, early nonzero exit rather than a deferred Settings() traceback.
    monkeypatch.delenv('SNOWTOOL_SNOWDB_CONFIG', raising=False)

    result = CliRunner().invoke(cli, ['api', 'serve', '--check'])

    assert result.exit_code != 0
    assert '--config' in result.output
