"""confirm_destructive: --yes bypass and the non-TTY refusal."""

import click
import pytest

from snowtool.cli._confirm import confirm_destructive


def test_yes_bypasses_everything():
    confirm_destructive('remove it?', yes=True)  # no prompt, no error


def test_non_tty_without_yes_refuses():
    # pytest's captured stdin is not a TTY, which is exactly the CI case.
    with pytest.raises(click.ClickException, match='--yes'):
        confirm_destructive('remove it?', yes=False)
