"""The destructive-operation gate, plus the shared removal-command flow.

``confirm_destructive`` prompts on a TTY and demands ``--yes`` elsewhere;
``run_removal`` builds the dry-run -> confirm -> remove -> echo shape shared
by ``dataset remove-date`` and ``pourpoint remove`` on top of it.
"""

from __future__ import annotations

import sys

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from collections.abc import Callable


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


def run_removal(
    label: str,
    prompt: str,
    *,
    preview: Callable[[], bool],
    execute: Callable[[], bool],
    dry_run: bool,
    yes: bool,
) -> None:
    """The shared dry-run -> confirm -> remove -> echo shape.

    ``dataset remove-date`` and ``pourpoint remove`` are otherwise identical:
    a dry run reports presence without deleting; a real run gates on
    :func:`confirm_destructive` then reports what happened. The two removal
    behaviours are passed as separate no-arg callables -- ``preview`` (the
    dry-run probe) and ``execute`` (the real deletion) -- each returning whether
    the target existed; the split keeps each side statically typed instead of a
    single ``Callable[..., bool]`` the call sites adapt with a ``dry_run`` kwarg.
    ``label`` names the target in every echoed line (e.g. ``'snodas
    2018-01-01'`` or a pourpoint triplet); ``prompt`` is the confirmation
    question (only shown for a real, non-``--yes`` removal).
    """
    if dry_run:
        present = preview()
        click.echo(f'would remove {label}' if present else f'{label}: absent')
        return

    confirm_destructive(prompt, yes=yes)

    if execute():
        click.echo(f'removed {label}')
    else:
        click.echo(f'{label}: absent (nothing removed)')
