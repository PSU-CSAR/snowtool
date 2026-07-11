"""The destructive-operation gate: prompt on a TTY, demand --yes elsewhere."""

from __future__ import annotations

import sys

import click


def confirm_destructive(prompt: str, *, yes: bool) -> None:
    """Gate an irreversible operation.

    ``--yes`` bypasses; an interactive stdin prompts (aborting on decline);
    a non-TTY stdin (scripts, CI) has no one to answer, so it fails with a
    pointer to ``--yes`` rather than hanging or proceeding silently.
    """
    if yes:
        return
    if not sys.stdin.isatty():
        raise click.ClickException(
            'stdin is not a TTY; pass --yes to proceed non-interactively.',
        )
    click.confirm(prompt, abort=True)
