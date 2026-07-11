"""The CLI's console pair and the root --color/--quiet options."""

from click.testing import CliRunner

from snowtool.cli import _console, cli


def test_out_is_stdout_err_is_stderr():
    # `.file` is a live property that reads `sys.stdout`/`sys.stderr` at access
    # time (rich.console.Console.file), so under pytest's default fd-capture
    # it resolves to the capture proxy rather than a real `<stdout>`/`<stderr>`
    # file object. `.stderr` is the stable, capture-independent marker of which
    # stream a console is bound to.
    assert _console.out().stderr is False
    assert _console.err().stderr is True


def test_configure_color_always_forces_terminal():
    _console.configure(color='always')
    try:
        assert _console.out().is_terminal is True
        assert _console.err().is_terminal is True
    finally:
        _console.configure()  # restore defaults for other tests


def test_configure_color_never_disables_terminal():
    _console.configure(color='never')
    try:
        assert _console.out().is_terminal is False
    finally:
        _console.configure()


def test_configure_quiet_silences_err_only():
    _console.configure(quiet=True)
    try:
        assert _console.err().quiet is True
        assert _console.out().quiet is False
    finally:
        _console.configure()


def test_root_options_are_accepted():
    runner = CliRunner()
    result = runner.invoke(cli, ['--color', 'never', '--quiet', '--version'])
    assert result.exit_code == 0


def test_non_terminal_console_gets_fixed_wide_width():
    # pytest capture is not a terminal, so both consoles built at import time
    # (and any rebuilt by configure()) must be widened to avoid rich's 80-col
    # non-TTY default folding wide table rows mid-word.
    assert _console.out().is_terminal is False
    assert _console.out().width == _console._NON_TERMINAL_WIDTH
    assert _console.err().width == _console._NON_TERMINAL_WIDTH


def test_configure_non_terminal_still_widens():
    _console.configure(color='auto')
    try:
        assert _console.out().width == _console._NON_TERMINAL_WIDTH
    finally:
        _console.configure()
