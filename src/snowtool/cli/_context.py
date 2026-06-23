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

from snowtool.snowdb.zone_layer_providers import DEFAULT_ZONE_LAYER_PROVIDERS

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from snowtool.snowdb.db import SnowDb
    from snowtool.snowdb.manager import SnowDbManager
    from snowtool.snowdb.zone_layer import ZoneLayerProvider, ZoneLayerSource


@dataclass
class CliContext:
    """The state the root ``cli`` group hands every command via ``ctx.obj``.

    Holds the ``--root`` value (or ``None`` to fall back to the ``snowdb_path``
    setting) and the zone-layer providers to bind, and lazily *opens* a single
    :class:`SnowDb` from its root config the first time :attr:`snowdb` is read --
    so the CLI serves exactly the registered datasets. ``zone_layer_sources``
    overrides a provider's default generation source by provider name (tests inject
    local files to avoid hitting 3DEP / the MRLC download); an unlisted provider
    keeps SnowDb's default source.
    """

    root: Path | None = None
    zone_layer_providers: tuple[ZoneLayerProvider, ...] = DEFAULT_ZONE_LAYER_PROVIDERS
    zone_layer_sources: dict[str, ZoneLayerSource] | None = None
    _snowdb: SnowDb | None = field(default=None, init=False, repr=False)

    @property
    def snowdb(self) -> SnowDb:
        """The invocation's SnowDb, opened once on first access.

        Resolves ``--root`` if given, otherwise the ``snowdb_path`` setting (read
        only here, so commands that never touch the database never require it), and
        opens the snowdb from the root config there.
        """
        if self._snowdb is None:
            from snowtool.settings import Settings
            from snowtool.snowdb.db import SnowDb

            root = self.root if self.root is not None else Settings().snowdb_path
            self._snowdb = SnowDb.open(
                root,
                zone_layer_providers=self.zone_layer_providers,
                zone_layer_sources=self.zone_layer_sources,
            )
        return self._snowdb

    @property
    def manager(self) -> SnowDbManager:
        """The invocation's :class:`SnowDbManager`, wrapping its read SnowDb.

        The write/admin seam: management commands take :func:`pass_manager` to get
        this, then mutate through it and read through ``manager.db``. It wraps the
        same lazily-opened :attr:`snowdb`, so opening still happens once.
        """
        from snowtool.snowdb.manager import SnowDbManager

        return SnowDbManager(self.snowdb)


def pass_snowdb[**P, R](
    f: Callable[Concatenate[SnowDb, P], R],
) -> Callable[P, R]:
    """Inject the invocation's :class:`SnowDb` as a command's first argument.

    Wraps :func:`click.pass_obj`: the wrapped callback receives the lazily-built
    SnowDb (from the :class:`CliContext` on ``ctx.obj``) instead of the context
    object, so command bodies talk to the domain API and never to click context.
    Opening can fail if the root has no config (never ``snowdb init``-ed); that is
    surfaced as a clean CLI error rather than a traceback.
    """

    @click.pass_obj
    @functools.wraps(f)
    def wrapper(ctx_obj: CliContext, /, *args: P.args, **kwargs: P.kwargs) -> R:
        import click as _click

        from snowtool.exceptions import SnowDbConfigError

        try:
            snowdb = ctx_obj.snowdb
        except SnowDbConfigError as e:
            raise _click.ClickException(str(e)) from e
        return f(snowdb, *args, **kwargs)

    return wrapper


def pass_manager[**P, R](
    f: Callable[Concatenate[SnowDbManager, P], R],
) -> Callable[P, R]:
    """Inject the invocation's :class:`SnowDbManager` as a command's first argument.

    The write-command counterpart to :func:`pass_snowdb`: management commands that
    mutate the database take this, then write through the manager and read through
    ``manager.db``. Opening can fail if the root has no config (never ``snowdb
    init``-ed); that is surfaced as a clean CLI error rather than a traceback.
    """

    @click.pass_obj
    @functools.wraps(f)
    def wrapper(ctx_obj: CliContext, /, *args: P.args, **kwargs: P.kwargs) -> R:
        import click as _click

        from snowtool.exceptions import SnowDbConfigError

        try:
            manager = ctx_obj.manager
        except SnowDbConfigError as e:
            raise _click.ClickException(str(e)) from e
        return f(manager, *args, **kwargs)

    return wrapper
