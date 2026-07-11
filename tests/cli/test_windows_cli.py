"""``snowtool windows`` -- Windows-only admin tooling.

Covers the whole ``windows`` group: its own ``add-to-path`` (backed by the
pure PATH-string/path-classification logic in ``snowtool.cli._path_env``)
plus the nested ``iis`` subcommand (backed by ``snowtool.cli._iis``'s pure
render/argv-building functions). None of this can be exercised end-to-end on
this (non-Windows) test runner, so these tests cover the pure layers plus the
CLI-level checks that hold on any platform: --help wiring, click's own
required-option validation, and the clean non-Windows guards (genuinely
exercised here, since this suite runs on macOS/Linux).
"""

import sys

from pathlib import Path, PureWindowsPath

import pytest

from click.testing import CliRunner

from snowtool.cli import cli
from snowtool.cli._iis.provisioning import (
    install_args,
    remove_args,
    run_powershell,
    snowdb_root,
    venv_root,
)
from snowtool.cli._iis.web_config import render_web_config
from snowtool.cli._path_env import (
    append_entry,
    contains_entry,
    is_user_specific_install,
    path_entries,
    user_specific_install_message,
    users_root,
)


def test_users_root_is_userprofile_parent():
    environ = {'USERPROFILE': r'C:\Users\alice'}

    assert users_root(environ) == PureWindowsPath(r'C:\Users')


def test_is_user_specific_install_true_under_users_root():
    root = PureWindowsPath(r'C:\Users')

    assert is_user_specific_install(
        PureWindowsPath(r'C:\Users\alice\.local\bin'),
        root,
    )


def test_is_user_specific_install_false_for_shared_location():
    root = PureWindowsPath(r'C:\Users')

    assert not is_user_specific_install(PureWindowsPath(r'C:\ProgramData\uv\bin'), root)


def test_user_specific_install_message_names_directory_and_reinstall_steps():
    message = user_specific_install_message(
        PureWindowsPath(r'C:\Users\alice\.local\bin'),
        PureWindowsPath(r'C:\Users'),
    )

    assert r'C:\Users\alice\.local\bin' in message
    assert 'UV_TOOL_DIR' in message
    assert 'UV_TOOL_BIN_DIR' in message
    assert 'uv tool install snowtool' in message


def test_path_entries_drops_empty_segments():
    assert path_entries(r'C:\a;;C:\b;') == [r'C:\a', r'C:\b']


def test_contains_entry_is_case_insensitive():
    value = r'C:\Windows\System32;C:\ProgramData\uv\bin'

    assert contains_entry(value, PureWindowsPath(r'c:\programdata\uv\bin'))
    assert not contains_entry(value, PureWindowsPath(r'C:\ProgramData\uv\other'))


def test_append_entry_adds_a_semicolon_separated_entry():
    assert append_entry(r'C:\a;C:\b', PureWindowsPath(r'C:\c')) == r'C:\a;C:\b;C:\c'


def test_append_entry_handles_empty_current_value():
    assert append_entry('', PureWindowsPath(r'C:\c')) == r'C:\c'


def test_render_web_config_embeds_python_exe_and_snowdb_config():
    content = render_web_config(Path('/opt/snowtool/bin/python'), Path('/etc/snowdb'))

    assert 'processPath="/opt/snowtool/bin/python"' in content
    assert 'arguments="-m snowtool api serve --port %HTTP_PLATFORM_PORT%"' in content
    assert (
        '<environmentVariable name="SNOWTOOL_SNOWDB_CONFIG" value="/etc/snowdb" />'
        in content
    )


def test_render_web_config_escapes_xml_significant_characters():
    content = render_web_config(Path('/opt/a & b/python'), Path('/etc/snowdb'))

    assert '/opt/a &amp; b/python' in content
    assert '/opt/a & b/python' not in content


def test_venv_root_is_python_exe_grandparent():
    assert venv_root(Path('/opt/tools/snowtool/Scripts/python.exe')) == Path(
        '/opt/tools/snowtool',
    )


def test_snowdb_root_passes_through_a_directory(tmp_path):
    assert snowdb_root(tmp_path) == tmp_path


def test_snowdb_root_resolves_a_config_file_to_its_parent(tmp_path):
    config_file = tmp_path / 'snowdb_conf.json'
    config_file.write_text('{}')

    assert snowdb_root(config_file) == tmp_path


def test_install_args_builds_expected_powershell_invocation():
    args = install_args(
        site_name='snowtool',
        physical_path=Path('/inetpub/snowtool'),
        venv_path=Path('/opt/tools/snowtool'),
        snowdb_path=Path('/data/snowdb'),
        hostname='snow.example.org',
        port=443,
        protocol='https',
        cert_thumbprint=None,
        recycle_time='03:00:00',
        access_log_dir=None,
    )

    assert args[:6] == [
        'powershell',
        '-NoProfile',
        '-NonInteractive',
        '-ExecutionPolicy',
        'Bypass',
        '-File',
    ]
    assert args[6].endswith('install_site.ps1')
    assert '-SiteName' in args
    assert 'snowtool' in args
    assert '-SnowdbPath' in args
    assert '/data/snowdb' in args
    assert '-Hostname' in args
    assert 'snow.example.org' in args
    assert '-CertThumbprint' not in args
    assert '-AccessLogDir' not in args


def test_install_args_includes_cert_thumbprint_and_access_log_dir_when_given():
    args = install_args(
        site_name='snowtool',
        physical_path=Path('/inetpub/snowtool'),
        venv_path=Path('/opt/tools/snowtool'),
        snowdb_path=Path('/data/snowdb'),
        hostname='snow.example.org',
        port=443,
        protocol='https',
        cert_thumbprint='ABCDEF0123456789',
        recycle_time='03:00:00',
        access_log_dir=Path('/var/log/snowtool'),
    )

    assert '-CertThumbprint' in args
    assert 'ABCDEF0123456789' in args
    assert '-AccessLogDir' in args
    assert '/var/log/snowtool' in args


def test_remove_args_builds_expected_powershell_invocation():
    args = remove_args(site_name='snowtool')

    assert args[:6] == [
        'powershell',
        '-NoProfile',
        '-NonInteractive',
        '-ExecutionPolicy',
        'Bypass',
        '-File',
    ]
    assert args[6].endswith('remove_site.ps1')
    assert args[-2:] == ['-SiteName', 'snowtool']


def test_run_powershell_invokes_injected_runner_with_check_true():
    calls = []

    def fake_runner(args, **kwargs):
        calls.append((args, kwargs))
        return 'result'

    result = run_powershell(['powershell', '-File', 'x.ps1'], runner=fake_runner)

    assert result == 'result'
    assert calls == [(['powershell', '-File', 'x.ps1'], {'check': True})]


def test_windows_help_lists_iis_and_add_to_path():
    result = CliRunner().invoke(cli, ['windows', '--help'])

    assert result.exit_code == 0
    assert 'iis' in result.output
    assert 'add-to-path' in result.output


@pytest.mark.skipif(sys.platform == 'win32', reason='windows group is visible on win32')
def test_windows_group_is_hidden_but_still_reachable_off_platform():
    # Genuinely exercised: this suite runs on macOS/Linux, so the group is
    # actually hidden here rather than being simulated.
    root_help = CliRunner().invoke(cli, ['--help'])
    assert root_help.exit_code == 0
    assert 'windows' not in root_help.output

    group_help = CliRunner().invoke(cli, ['windows', '--help'])
    assert group_help.exit_code == 0


def test_add_to_path_fails_cleanly_on_non_windows():
    # Genuinely exercised: this suite runs on macOS/Linux, so the platform
    # guard fires for real here rather than being simulated.
    result = CliRunner().invoke(cli, ['windows', 'add-to-path'])

    assert result.exit_code != 0
    assert 'Windows' in result.output


def test_iis_help_lists_install_and_remove():
    result = CliRunner().invoke(cli, ['windows', 'iis', '--help'])

    assert result.exit_code == 0
    assert 'install' in result.output
    assert 'remove' in result.output


def test_install_requires_hostname(tmp_path):
    result = CliRunner().invoke(
        cli,
        [
            'windows',
            'iis',
            'install',
            str(tmp_path / 'site'),
            '--config',
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert '--hostname' in result.output


def test_install_requires_config(monkeypatch, tmp_path):
    monkeypatch.delenv('SNOWTOOL_SNOWDB_CONFIG', raising=False)

    result = CliRunner().invoke(
        cli,
        [
            'windows',
            'iis',
            'install',
            str(tmp_path / 'site'),
            '--hostname',
            'snow.example.org',
        ],
    )

    assert result.exit_code != 0
    assert '--config' in result.output


def test_install_fails_cleanly_on_non_windows(tmp_path):
    # Genuinely exercised: this suite runs on macOS/Linux, so the platform
    # guard fires for real here rather than being simulated.
    result = CliRunner().invoke(
        cli,
        [
            'windows',
            'iis',
            'install',
            str(tmp_path / 'site'),
            '--hostname',
            'snow.example.org',
            '--config',
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert 'Windows' in result.output


def test_remove_fails_cleanly_on_non_windows(tmp_path):
    result = CliRunner().invoke(
        cli,
        ['windows', 'iis', 'remove', str(tmp_path / 'site')],
    )

    assert result.exit_code != 0
    assert 'Windows' in result.output
