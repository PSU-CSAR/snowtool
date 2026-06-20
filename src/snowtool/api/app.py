import logging

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from snowtool.settings import Settings

from .exceptions import APIError, api_error_handler
from .request_state import RequestState
from .routers.root import router as root_router
from .tags import Tags


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[RequestState]:
    async with RequestState(app, app.state.settings) as request_state:
        yield request_state


def get_app(
    settings: Settings | None = None,
    logger: logging.Logger | None = None,
) -> FastAPI:
    if settings is None:
        settings = Settings()

    if logger is None:
        logger = logging.getLogger(__name__)

    app: FastAPI = FastAPI(
        title='My FastAPI skeleton project',
        lifespan=lifespan,
        openapi_tags=Tags.metadata(),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    app.state.settings = settings
    app.state.logger = logger

    app.include_router(root_router)

    app.exception_handler(APIError)(api_error_handler)

    return app
