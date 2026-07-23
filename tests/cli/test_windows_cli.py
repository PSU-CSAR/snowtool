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
    base_python_root,
    install_args,
    remove_args,
    run_powershell,
    snowdb_root,
    venv_root,
)
from snowtool.cli._iis.web_config import rasterio_data_env, render_web_config
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
    assert 'UV_PYTHON_INSTALL_DIR' in message
    assert 'UV_LINK_MODE' in message
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


def test_render_web_config_pins_data_env_vars():
    content = render_web_config(
        Path('/opt/snowtool/bin/python'),
        Path('/etc/snowdb'),
        data_env={
            'GDAL_DATA': Path('/opt/rasterio/gdal_data'),
            'PROJ_DATA': Path('/opt/rasterio/proj_data'),
        },
    )

    assert (
        '<environmentVariable name="GDAL_DATA" value="/opt/rasterio/gdal_data" />'
        in content
    )
    assert (
        '<environmentVariable name="PROJ_DATA" value="/opt/rasterio/proj_data" />'
        in content
    )


def test_rasterio_data_env_pins_all_proj_spellings_to_the_wheel():
    import rasterio

    env = rasterio_data_env()
    rasterio_package = Path(rasterio.__file__).parent

    assert set(env) == {'GDAL_DATA', 'PROJ_DATA', 'PROJ_LIB'}
    assert env['PROJ_DATA'] == env['PROJ_LIB'] == rasterio_package / 'proj_data'
    assert env['GDAL_DATA'] == rasterio_package / 'gdal_data'
    assert all(path.is_dir() for path in env.values())


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


def test_base_python_root_is_the_base_prefix_only_inside_a_venv():
    base = '/uv/python/cpython-3.14'

    assert base_python_root('/opt/tools/snowtool', base) == Path(base)
    assert base_python_root(base, base) is None


def test_install_args_builds_expected_powershell_invocation():
    args = install_args(
        site_name='snowtool',
        physical_path=Path('/inetpub/snowtool'),
        venv_path=Path('/opt/tools/snowtool'),
        base_python_path=None,
        snowdb_path=Path('/data/snowdb'),
        hostname='snow.example.org',
        port=443,
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
    protocol_index = args.index('-Protocol')
    assert args[protocol_index + 1] == 'https'
    assert '-BasePythonPath' not in args
    assert '-CertThumbprint' not in args
    assert '-AccessLogDir' not in args


def test_install_args_includes_optional_paths_and_thumbprint_when_given():
    args = install_args(
        site_name='snowtool',
        physical_path=Path('/inetpub/snowtool'),
        venv_path=Path('/opt/tools/snowtool'),
        base_python_path=Path('/opt/uv/python/cpython-3.14'),
        snowdb_path=Path('/data/snowdb'),
        hostname='snow.example.org',
        port=443,
        cert_thumbprint='ABCDEF0123456789',
        recycle_time='03:00:00',
        access_log_dir=Path('/var/log/snowtool'),
    )

    assert '-BasePythonPath' in args
    assert '/opt/uv/python/cpython-3.14' in args
    assert '-CertThumbprint' in args
    assert 'ABCDEF0123456789' in args
    assert '-AccessLogDir' in args
    assert '/var/log/snowtool' in args


def test_remove_args_builds_expected_powershell_invocation():
    args = remove_args(
        site_name='snowtool',
        venv_path=Path('/opt/tools/snowtool'),
        base_python_path=Path('/opt/uv/python/cpython-3.14'),
        snowdb_path=Path('/data/snowdb'),
        physical_path=Path('/inetpub/snowtool'),
    )

    assert args[:6] == [
        'powershell',
        '-NoProfile',
        '-NonInteractive',
        '-ExecutionPolicy',
        'Bypass',
        '-File',
    ]
    assert args[6].endswith('remove_site.ps1')
    assert args[7:] == [
        '-SiteName',
        'snowtool',
        '-VenvPath',
        '/opt/tools/snowtool',
        '-SnowdbPath',
        '/data/snowdb',
        '-PhysicalPath',
        '/inetpub/snowtool',
        '-BasePythonPath',
        '/opt/uv/python/cpython-3.14',
    ]


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


@pytest.mark.parametrize(
    'argv',
    [
        pytest.param(lambda tmp: ['windows', 'add-to-path'], id='add-to-path'),
        pytest.param(
            lambda tmp: [
                'windows',
                'iis',
                'install',
                str(tmp / 'site'),
                '--hostname',
                'snow.example.org',
                '--config',
                str(tmp),
            ],
            id='iis-install',
        ),
        pytest.param(
            lambda tmp: ['windows', 'iis', 'remove', str(tmp / 'site'), '-C', str(tmp)],
            id='iis-remove',
        ),
    ],
)
def test_windows_only_command_fails_cleanly_on_non_windows(tmp_path, argv):
    # Genuinely exercised: this suite runs on macOS/Linux, so the platform guard
    # fires for real here (past option parsing) rather than being simulated.
    result = CliRunner().invoke(cli, argv(tmp_path))

    assert result.exit_code != 0
    assert 'Windows' in result.output


def test_iis_help_lists_install_and_remove():
    result = CliRunner().invoke(cli, ['windows', 'iis', '--help'])

    assert result.exit_code == 0
    assert 'install' in result.output
    assert 'remove' in result.output


def _install_argv(tmp_path, *, hostname='snow.example.org', config=True, extra=()):
    """The `windows iis install` argv, with the common options pre-wired.

    ``hostname``/``config`` drop their option when set falsy (to exercise click's
    required-option validation); ``extra`` appends trailing flags (``--only``,
    the removed ``--protocol``, ...).
    """
    argv = ['windows', 'iis', 'install', str(tmp_path / 'site')]
    if hostname:
        argv += ['--hostname', hostname]
    if config:
        argv += ['--config', str(tmp_path)]
    return [*argv, *extra]


def test_install_requires_hostname(tmp_path):
    result = CliRunner().invoke(cli, _install_argv(tmp_path, hostname=None))

    assert result.exit_code != 0
    assert '--hostname' in result.output


def test_install_requires_config(monkeypatch, tmp_path):
    monkeypatch.delenv('SNOWTOOL_SNOWDB_CONFIG', raising=False)

    result = CliRunner().invoke(cli, _install_argv(tmp_path, config=False))

    assert result.exit_code != 0
    assert '--config' in result.output


def test_install_help_documents_only_option():
    result = CliRunner().invoke(cli, ['windows', 'iis', 'install', '--help'])

    assert result.exit_code == 0
    assert '--only' in result.output
    assert '--skip-site' not in result.output
    assert '--skip-config' not in result.output


@pytest.mark.parametrize('only', ['config', 'site'])
def test_install_accepts_only_config_or_site(tmp_path, only):
    # Genuinely exercised past option parsing: the non-Windows guard fires
    # only after click has validated --only, so an invalid choice is
    # distinguishable here from a valid one hitting require_windows().
    result = CliRunner().invoke(cli, _install_argv(tmp_path, extra=['--only', only]))

    assert result.exit_code != 0
    assert 'Windows' in result.output


def test_install_rejects_invalid_only_value(tmp_path):
    result = CliRunner().invoke(
        cli,
        _install_argv(tmp_path, extra=['--only', 'bogus']),
    )

    assert result.exit_code == 2  # click Choice rejection


def test_install_rejects_a_removed_flag(tmp_path):
    # One representative removed flag (--protocol was dropped when https became
    # the only supported binding; --skip-site/--skip-config were likewise
    # removed in favor of --only). click rejects any of them the same way.
    result = CliRunner().invoke(
        cli,
        _install_argv(tmp_path, extra=['--protocol', 'http']),
    )

    assert result.exit_code == 2
    assert 'No such option' in result.output
