"""Domain-exception -> RFC 9457 problem+json handlers.

gazebo wires the ``ProblemException``, ``RequestValidationError`` (->422), and
``ParamError`` (->400) handlers for free. We register handlers only for the
*specific* domain exceptions we deliberately convert from a 500 into a client
error -- never a bare ``ValueError``/``FileNotFoundError``, so a genuine server bug
still surfaces as a 500:

* :class:`PourpointCoverageError` -> 409 (the AOI is not covered by the dataset grid)
* :class:`PourpointNotFoundError` -> 404 (no stored AOI record for the triplet)
* :class:`AOIRasterNotFoundError` -> 404 (the AOI raster has not been built)
* :class:`QueryParameterError` -> 422 (unknown variable/zone, runaway cross)
"""

from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING

from fastapi import Response
from gazebo.problems import ProblemDetail
from gazebo.rels import MediaType

from snowtool.exceptions import (
    AOIRasterNotFoundError,
    PourpointCoverageError,
    PourpointNotFoundError,
    QueryParameterError,
)

if TYPE_CHECKING:
    from fastapi import FastAPI, Request

    from snowtool.exceptions import SNODASError


def _problem_response(status: int, detail: str) -> Response:
    problem = ProblemDetail(
        title=HTTPStatus(status).phrase,
        status=status,
        detail=detail,
    )
    return Response(
        content=problem.model_dump_json(),
        status_code=status,
        media_type=MediaType.PROBLEM,
    )


def _handler(status: HTTPStatus):
    async def handle(request: Request, exc: SNODASError) -> Response:
        return _problem_response(status, str(exc))

    return handle


def install_exception_handlers(app: FastAPI) -> None:
    """Register the domain-exception -> problem+json handlers on ``app``."""
    app.add_exception_handler(PourpointCoverageError, _handler(HTTPStatus.CONFLICT))  # type: ignore[arg-type]
    app.add_exception_handler(PourpointNotFoundError, _handler(HTTPStatus.NOT_FOUND))  # type: ignore[arg-type]
    app.add_exception_handler(AOIRasterNotFoundError, _handler(HTTPStatus.NOT_FOUND))  # type: ignore[arg-type]
    app.add_exception_handler(
        QueryParameterError,
        _handler(HTTPStatus.UNPROCESSABLE_ENTITY),
    )  # type: ignore[arg-type]
