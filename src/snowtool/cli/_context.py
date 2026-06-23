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
    from snowtool.snowdb.spec import DatasetSpec
    from snowtool.snowdb.zone_layer import ZoneLayerProvider, ZoneLayerSource


@dataclass
class CliContext:
    """The state the root ``cli`` group hands every command via ``ctx.obj``.

    Holds the ``--root`` value (or ``None`` to fall back to the ``snowdb_path``
    setting), the zone-layer providers to bind, and lazily builds a single
    :class:`SnowDb` the first time :attr:`snowdb` is read. By default the database
    is *opened from its root config* -- it serves exactly the registered datasets.
    ``specs`` is an escape hatch: when supplied (tests, programmatic use) the
    SnowDb is built by injecting those specs instead of following the config
    links, so a synthetic dataset needs no on-disk registration.
    ``zone_layer_sources`` overrides a provider's default generation source by
    provider name (tests inject local files to avoid hitting 3DEP / the MRLC
    download); an unlisted provider keeps SnowDb's default source.
    """

    root: Path | None = None
    specs: tuple[DatasetSpec, ...] | None = None
    zone_layer_providers: tuple[ZoneLayerProvider, ...] = DEFAULT_ZONE_LAYER_PROVIDERS
    zone_layer_sources: dict[str, ZoneLayerSource] | None = None
    _snowdb: SnowDb | None = field(default=None, init=False, repr=False)

    @property
    def snowdb(self) -> SnowDb:
        """The invocation's SnowDb, built once on first access.

        Resolves ``--root`` if given, otherwise the ``snowdb_path`` setting (read
        only here, so commands that never touch the database never require it).
        Opens from the root config unless ``specs`` were injected, in which case it
        binds those directly.
        """
        if self._snowdb is None:
            from snowtool.settings import Settings
            from snowtool.snowdb.db import SnowDb

            root = self.root if self.root is not None else Settings().snowdb_path
            if self.specs is None:
                self._snowdb = SnowDb.open(
                    root,
                    zone_layer_providers=self.zone_layer_providers,
                    zone_layer_sources=self.zone_layer_sources,
                )
            else:
                self._snowdb = SnowDb(
                    root,
                    self.specs,
                    zone_layer_providers=self.zone_layer_providers,
                    zone_layer_sources=self.zone_layer_sources,
                )
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
