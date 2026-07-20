"""Per-invocation CLI context and the lazy ``@pass_snowdb`` decorator.

The CLI is a thin shell over the snowdb Python API: a command resolves a
:class:`~snowtool.snowdb.db.SnowDb`, calls a domain method, and renders. To keep
that one SnowDb-per-invocation without a global factory, the root ``cli`` group's
callback stores a :class:`CliContext` on ``ctx.obj``; commands that need the
database take :func:`pass_snowdb`, which builds (and caches) it on first use.

The build is *lazy* on purpose: ``--version``, ``--help``, and the ``api`` group
must never construct a SnowDb -- doing so would require a ``--config`` for commands
that have no business touching the database.
"""

from __future__ import annotations

import functools

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Concatenate

import click

from snowtool.snowdb.zones.zone_layer_providers import DEFAULT_ZONE_LAYER_PROVIDERS

if TYPE_CHECKING:
    from collections.abc import Callable

    from snowtool.snowdb.db import SnowDb
    from snowtool.snowdb.manager import SnowDbManager
    from snowtool.snowdb.zones.zone_layer import ZoneLayerProvider


@dataclass
class CliContext:
    """The state the root ``cli`` group hands every command via ``ctx.obj``.

    Holds the resolved ``--config`` root (set by :data:`config_option`, which reads
    the flag or its ``SNOWTOOL_SNOWDB_CONFIG`` env var) and the zone-layer providers
    to bind, and lazily *opens* a single :class:`SnowDb` from its root config the
    first time :attr:`snowdb` is read -- so the CLI serves exactly the registered
    datasets. Generation sources are declared in the config (``RootConfig.sources``),
    not injected here. The CLI depends on no pydantic ``Settings``: that is an API
    concern (see :mod:`snowtool.api.settings`).
    """

    config: Path | None = None
    zone_layer_providers: tuple[ZoneLayerProvider, ...] = DEFAULT_ZONE_LAYER_PROVIDERS
    _snowdb: SnowDb | None = field(default=None, init=False, repr=False)

    @property
    def snowdb(self) -> SnowDb:
        """The invocation's SnowDb, opened once on first access.

        Opens the snowdb from :attr:`config` (the ``--config`` flag or its
        ``SNOWTOOL_SNOWDB_CONFIG`` env var). With neither set, raises a clean CLI
        error rather than a traceback.
        """
        if self._snowdb is None:
            from snowtool.snowdb.db import SnowDb

            if self.config is None:
                raise click.ClickException(
                    'No snowdb configured. Pass --config/-C or set '
                    'SNOWTOOL_SNOWDB_CONFIG.',
                )
            self._snowdb = SnowDb.open(
                self.config,
                zone_layer_providers=self.zone_layer_providers,
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


def _inject[T, **P, R](
    f: Callable[Concatenate[T, P], R],
    build: Callable[[CliContext], T],
) -> Callable[P, R]:
    """Wrap ``f`` so it receives ``build(ctx_obj)`` as its first argument.

    The shared scaffold behind :func:`pass_snowdb`/:func:`pass_manager`: it wraps
    :func:`click.pass_obj` and builds the target from the :class:`CliContext` on
    ``ctx.obj`` (lazily opening the database). Errors need no handling here: the
    :class:`~snowtool.exceptions.SnowDbConfigError` an un-initialized root raises
    is a ``SnowtoolError``, which the root group maps to a clean
    ``ClickException`` for every command.
    """

    @click.pass_obj
    @functools.wraps(f)
    def wrapper(ctx_obj: CliContext, /, *args: P.args, **kwargs: P.kwargs) -> R:
        return f(build(ctx_obj), *args, **kwargs)

    return wrapper


def pass_snowdb[**P, R](
    f: Callable[Concatenate[SnowDb, P], R],
) -> Callable[P, R]:
    """Inject the invocation's :class:`SnowDb` as a command's first argument.

    The wrapped callback receives the lazily-built SnowDb (from the
    :class:`CliContext` on ``ctx.obj``) instead of the context object, so command
    bodies talk to the domain API and never to click context.
    """
    return _inject(f, lambda ctx_obj: ctx_obj.snowdb)


def pass_manager[**P, R](
    f: Callable[Concatenate[SnowDbManager, P], R],
) -> Callable[P, R]:
    """Inject the invocation's :class:`SnowDbManager` as a command's first argument.

    The write-command counterpart to :func:`pass_snowdb`: management commands that
    mutate the database take this, then write through the manager and read through
    ``manager.db``.
    """
    return _inject(f, lambda ctx_obj: ctx_obj.manager)


def _apply_config(
    ctx: click.Context,
    param: click.Parameter,
    value: Path | None,
) -> Path | None:
    """Stash a command's ``--config`` on the :class:`CliContext`.

    The callback behind :data:`config_option`: it runs during arg parsing (before
    the command body), so ``pass_snowdb``/``pass_manager`` and ``init`` read
    the resolved config off ``ctx.obj``. ``value`` is the flag or its
    ``SNOWTOOL_SNOWDB_CONFIG`` env var (click resolves the envvar); a bare
    ``None`` (neither set) leaves any injected :class:`CliContext` untouched.
    """
    if value is not None:
        ctx.ensure_object(CliContext).config = value
    return value


# The per-command snowdb-selection option. Applied only to commands that open a
# snowdb, so a command that doesn't (``api serve``, ``--version``) carries no
# config flag -- the dependency is declared exactly where it exists. Config-less
# (``expose_value=False``): it sets ``CliContext.config`` via the callback rather
# than adding a parameter to every command body.
config_option = click.option(
    '--config',
    '-C',
    type=click.Path(path_type=Path),
    default=None,
    envvar='SNOWTOOL_SNOWDB_CONFIG',
    expose_value=False,
    callback=_apply_config,
    help='Snowdb config file or its directory '
    '(defaults to the SNOWTOOL_SNOWDB_CONFIG env var).',
)
