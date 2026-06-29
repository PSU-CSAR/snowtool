"""The FastAPI app, wired with gazebo DI / deferred links / problem responses.

``get_app`` builds one catalog :class:`SnowDb` (cache-free) from the configured
root and registers it -- with :class:`Settings` and the app-scoped
:class:`SnowDbReader` -- as gazebo providers. Catalog-only routes (``/``,
``/datasets``, ``/aois``) inject ``SnowDb``; the stats routes inject
``SnowDbReader`` (its loop-affine COG cache is born in the app's event loop at
lifespan). The per-dataset stats routers are registered by looping the catalog's
datasets, so each dataset's generated response model surfaces a precise OpenAPI
schema.

There is no module-level ``app``: the ASGI server builds it via the factory
(``uvicorn snowtool.api.app:get_app --factory``), so importing this module has no
side effects and needs no config. ``get_app`` opens the catalog when *called* (at
server start), which is when ``SNOWTOOL_SNOWDB_CONFIG`` must be set; tests call
``get_app(settings=...)`` directly.
"""

from __future__ import annotations

import logging

from gazebo.di import Providers
from gazebo.ext.fastapi import GazeboApp

from snowtool.settings import Settings
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.reader import SnowDbReader

from .exceptions import install_exception_handlers
from .routers.aois import router as aois_router
from .routers.datasets import router as datasets_router
from .routers.root import API_DESCRIPTION, API_TITLE
from .routers.root import router as root_router
from .routers.stats import build_stats_router
from .tags import Tags


def get_app(
    settings: Settings | None = None,
    logger: logging.Logger | None = None,
) -> GazeboApp:
    if settings is None:
        settings = Settings()

    if logger is None:
        logger = logging.getLogger(__name__)

    # One immutable catalog for the app's lifetime: used directly to enumerate
    # datasets (and register per-dataset stats routes) and registered as the
    # app-scoped SnowDb provider so routes inject the same instance.
    catalog = SnowDb.open(settings.snowdb_config)

    providers = Providers()
    providers.app(Settings, lambda: settings)
    providers.app(SnowDb, lambda: catalog)
    providers.app(SnowDbReader)  # built from SnowDb + Settings via __provide__

    # CORS is off by default (GazeboApp accepts cors= when a policy is wanted);
    # the previous permissive-but-contradictory config was dropped rather than
    # carried forward.
    app = GazeboApp(
        providers,
        title=API_TITLE,
        description=API_DESCRIPTION,
        openapi_tags=Tags.metadata(),
    )

    app.state.settings = settings
    app.state.logger = logger

    install_exception_handlers(app)

    app.include_router(root_router)
    app.include_router(datasets_router)
    app.include_router(aois_router)
    for name in catalog.datasets:
        app.include_router(build_stats_router(catalog[name]))

    return app
