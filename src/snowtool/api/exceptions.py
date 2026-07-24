"""Domain-exception -> RFC 9457 problem+json handlers.

gazebo wires the ``ProblemException``, ``RequestValidationError`` (->422), and
``ParamError`` (->400) handlers for free. We register handlers only for the
*specific* domain exceptions we deliberately convert from a 500 into a client
error -- never a bare ``ValueError``/``FileNotFoundError``, so a genuine server bug
still surfaces as a 500; :class:`IncompleteDatasetDataError` is mapped but stays
500-class (server data integrity, not a client mistake). Each maps onto a
registered :class:`ProblemType` (see ``problems.py``) so the response carries a
stable, resolvable ``type`` URI -- see the ``handlers`` mapping below for the
exception -> status pairing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Response
from gazebo.rels import MediaType

from snowtool.api import problems
from snowtool.exceptions import (
    AOIRasterNotFoundError,
    IncompleteDatasetDataError,
    PourpointCoverageError,
    PourpointNotFoundError,
    QueryParameterError,
    UnknownDatasetError,
)

if TYPE_CHECKING:
    from fastapi import FastAPI, Request
    from gazebo.problems import ProblemType

    from snowtool.exceptions import SnowtoolError


def _handler(problem_type: ProblemType):
    async def handle(request: Request, exc: SnowtoolError) -> Response:
        problem = problem_type.problem(detail=str(exc))
        return Response(
            content=problem.model_dump_json(),
            status_code=problem_type.status,
            media_type=MediaType.PROBLEM,
        )

    return handle


def install_exception_handlers(app: FastAPI) -> None:
    """Register the domain-exception -> problem+json handlers on ``app``."""
    handlers: dict[type[SnowtoolError], ProblemType] = {
        PourpointCoverageError: problems.POURPOINT_NOT_COVERED,
        PourpointNotFoundError: problems.POURPOINT_NOT_FOUND,
        AOIRasterNotFoundError: problems.AOI_RASTER_NOT_FOUND,
        UnknownDatasetError: problems.DATASET_NOT_FOUND,
        QueryParameterError: problems.INVALID_QUERY_PARAMETER,
        IncompleteDatasetDataError: problems.INCOMPLETE_DATASET_DATA,
    }
    for exc_type, problem_type in handlers.items():
        app.add_exception_handler(exc_type, _handler(problem_type))
