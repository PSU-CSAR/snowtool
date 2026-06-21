"""Per-invocation CLI context and the lazy ``@pass_snowdb`` decorator.

The CLI is a thin shell over the snowdb Python API: a command resolves a
:class:`~snowtool.snowdb.db.SnowDb`, calls a domain method, and renders. To keep
that one SnowDb-per-invocation without a global factory, the root ``cli`` group's
callback stores a :class:`CliContext` on ``ctx.obj``; commands that need the
database take :func:`pass_snowdb`, which builds (and caches) it on first use.

The build is *lazy* on purpose: ``version``, ``migration`` (path-only), and
``--help`` must never construct a SnowDb -- doing so would require the
``snowdb_path`` setting and emit an "uninitialized" warning for commands that
have no business touching the database.
"""

from __future__ import annotations

import functools

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Concatenate

import click

from snowtool.snowdb.datasets import DEFAULT_DATASET_SPECS

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from snowtool.snowdb.db import SnowDb
    from snowtool.snowdb.dem_source import DemSource
    from snowtool.snowdb.spec import DatasetSpec


@dataclass
class CliContext:
    """The state the root ``cli`` group hands every command via ``ctx.obj``.

    Holds the ``--root`` value (or ``None`` to fall back to the ``snowdb_path``
    setting) and the dataset specs to bind, and lazily builds a single
    :class:`SnowDb` the first time :attr:`snowdb` is read. ``dem_source`` overrides
    the database's default terrain source (tests inject a local one to avoid
    hitting 3DEP); ``None`` leaves SnowDb's default in place.
    """

    root: Path | None = None
    specs: tuple[DatasetSpec, ...] = DEFAULT_DATASET_SPECS
    dem_source: DemSource | None = None
    _snowdb: SnowDb | None = field(default=None, init=False, repr=False)

    @property
    def snowdb(self) -> SnowDb:
        """The invocation's SnowDb, built once on first access.

        Resolves ``--root`` if given, otherwise the ``snowdb_path`` setting; the
        setting is only read here, so commands that never touch the database
        never require it.
        """
        if self._snowdb is None:
            from snowtool.settings import Settings
            from snowtool.snowdb.db import SnowDb

            root = self.root if self.root is not None else Settings().snowdb_path
            self._snowdb = SnowDb(root, self.specs, dem_source=self.dem_source)
        return self._snowdb


def pass_snowdb[**P, R](
    f: Callable[Concatenate[SnowDb, P], R],
) -> Callable[P, R]:
    """Inject the invocation's :class:`SnowDb` as a command's first argument.

    Wraps :func:`click.pass_obj`: the wrapped callback receives the lazily-built
    SnowDb (from the :class:`CliContext` on ``ctx.obj``) instead of the context
    object, so command bodies talk to the domain API and never to click context.
    """

    @click.pass_obj
    @functools.wraps(f)
    def wrapper(ctx_obj: CliContext, /, *args: P.args, **kwargs: P.kwargs) -> R:
        return f(ctx_obj.snowdb, *args, **kwargs)

    return wrapper
