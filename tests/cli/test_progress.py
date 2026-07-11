"""RichProgress: the CLI binding of the domain ProgressReporter seam."""

import io

from rich.console import Console

from snowtool.cli._progress import RichProgress


def _console(force_terminal: bool) -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, stderr=True, force_terminal=force_terminal), buf


def test_non_tty_announces_label_once_and_advances_silently():
    console, buf = _console(force_terminal=False)
    with RichProgress(console=console).track('rasterizing', total=3) as task:
        task.advance()
        task.advance(2)
    assert buf.getvalue() == 'rasterizing...\n'


def test_tty_renders_live_progress():
    console, buf = _console(force_terminal=True)
    with RichProgress(console=console).track('rasterizing', total=2) as task:
        task.advance(2)
    assert 'rasterizing' in buf.getvalue()  # live bar rendered ANSI output


def test_prefix_prepends_to_label():
    console, buf = _console(force_terminal=False)
    with RichProgress(prefix='snodas ingest: ', console=console).track('dates'):
        pass
    assert buf.getvalue() == 'snodas ingest: dates...\n'


def test_indeterminate_total_still_tracks():
    console, buf = _console(force_terminal=True)
    with RichProgress(console=console).track('downloading') as task:
        task.advance()
    assert 'downloading' in buf.getvalue()
