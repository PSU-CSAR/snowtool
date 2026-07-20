"""The FastAPI app, wired with gazebo DI / deferred links / problem responses.

``get_app`` builds one catalog :class:`SnowDb` (cache-free) from the configured
root and registers it -- with :class:`Settings` and the app-scoped
:class:`SnowDbReader` -- as gazebo providers. Catalog-only routes (``/``,
``/datasets``, ``/pourpoints``) inject ``SnowDb``; the stats routes inject
``SnowDbReader`` (its loop-affine COG cache is born in the app's event loop at
lifespan). The stats routes are one generic router (``{dataset}`` a path param)
serving a single generic response schema, injected with ``SnowDbReader``.

There is no module-level ``app``: the ASGI server builds it via the factory
(``uvicorn snowtool.api.app:get_app --factory``), so importing this module has no
side effects and needs no config. ``get_app`` opens the catalog when *called* (at
server start), which is when ``SNOWTOOL_SNOWDB_CONFIG`` must be set; tests call
``get_app(settings=...)`` directly.
"""

from __future__ import annotations

from gazebo.di import Providers
from gazebo.ext.fastapi import GazeboApp
from gazebo.tags import Tag, tags_metadata

from snowtool.api.problems import MALFORMED_QUERY_PARAMETER
from snowtool.api.settings import Settings
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.raster.tiff_cache import TiffCache
from snowtool.snowdb.reader import SnowDbReader

from .exceptions import install_exception_handlers
from .routers.datasets import router as datasets_router
from .routers.pourpoints import router as pourpoints_router
from .routers.root import API_DESCRIPTION, API_TITLE
from .routers.root import router as root_router
from .routers.stats import router as stats_router
from .tags import Tags


def _provide_reader(db: SnowDb, settings: Settings) -> SnowDbReader:
    """The gazebo recipe for the app-scoped :class:`SnowDbReader`.

    The API owns the ``Settings``->domain translation: the reader takes plain
    read-path knobs, and this recipe sizes them from settings -- the COG cache from
    ``tiff_cache_size`` and the crossed-stats cap from ``max_zone_cells``. gazebo
    resolves ``db``/``settings`` from their app bindings and calls this inside the
    app's event loop at lifespan, so the reader's loop-affine cache is born there.
    (``snowdb`` itself imports no ``Settings``.)
    """
    return SnowDbReader(
        db,
        TiffCache(settings.tiff_cache_size),
        max_zone_cells=settings.max_zone_cells,
    )


def get_app(
    settings: Settings | None = None,
) -> GazeboApp:
    if settings is None:
        settings = Settings()

    # One immutable catalog for the app's lifetime: used directly to enumerate
    # datasets and registered as the app-scoped SnowDb provider so routes inject
    # the same instance.
    catalog = SnowDb.open(settings.snowdb_config)

    providers = Providers()
    providers.app(Settings, lambda: settings)
    providers.app(SnowDb, lambda: catalog)
    # Supplied recipe rather than a __provide__ on the reader (which is domain code
    # and imports no Settings); see _provide_reader.
    providers.app(SnowDbReader, _provide_reader)

    # CORS is off by default (GazeboApp accepts cors= when a policy is wanted).
    # ``query_problem`` gives gazebo's own malformed-query-parameter 400s a resolvable
    # ``type`` from our catalog instead of ``about:blank`` (see problems.py).
    app = GazeboApp(
        providers,
        title=API_TITLE,
        description=API_DESCRIPTION,
        openapi_tags=tags_metadata(*(Tag(name=member) for member in Tags)),
        query_problem=MALFORMED_QUERY_PARAMETER,
    )

    install_exception_handlers(app)

    app.include_router(root_router)
    app.include_router(datasets_router)
    app.include_router(pourpoints_router)
    app.include_router(stats_router)

    return app
