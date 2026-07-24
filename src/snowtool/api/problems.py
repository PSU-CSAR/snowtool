"""The API's catalog of RFC 9457 problem types, with stable ``type`` URIs.

Every problem this service raises is defined once here, so its ``type`` URI stops
defaulting to ``about:blank`` and stays stable and resolvable. The URI is the
service-relative ``/problems/{key}`` path served by the root router's ``get_problem``
route -- a client that receives a ``type`` can dereference it (or read the whole
``/problems`` catalog) to recover its meaning.

``exceptions.py`` maps the domain exceptions onto these types in its handlers; route
bodies that raise directly (e.g. an unknown dataset/problem key) call
``PROBLEMS[key].exception(...)``.
"""

from __future__ import annotations

from gazebo.problems import ProblemRegistry, ProblemType

PROBLEMS = ProblemRegistry()


def _define(key: str, *, title: str, status: int) -> ProblemType:
    # The type URI is the route that serves this very entry (see root.get_problem).
    return PROBLEMS.define(key, type=f'/problems/{key}', title=title, status=status)


POURPOINT_NOT_COVERED = _define(
    'pourpoint-not-covered',
    title='Pourpoint not covered by dataset grid',
    status=409,
)
POURPOINT_NOT_FOUND = _define(
    'pourpoint-not-found',
    title='Pourpoint not found',
    status=404,
)
AOI_RASTER_NOT_FOUND = _define(
    'aoi-raster-not-found',
    title='AOI raster not built',
    status=404,
)
INVALID_QUERY_PARAMETER = _define(
    'invalid-query-parameter',
    title='Invalid query parameter',
    status=422,
)
MALFORMED_QUERY_PARAMETER = _define(
    'malformed-query-parameter',
    # gazebo's schema-layer counterpart of the handler-raised
    # INVALID_QUERY_PARAMETER (422); wired via GazeboApp's ``query_problem=``.
    title='Malformed query parameter',
    status=400,
)
DATASET_NOT_FOUND = _define(
    'dataset-not-found',
    title='Dataset not found',
    status=404,
)
PROBLEM_TYPE_NOT_FOUND = _define(
    'problem-type-not-found',
    title='Problem type not found',
    status=404,
)
INCOMPLETE_DATASET_DATA = _define(
    'incomplete-dataset-data',
    title='Incomplete dataset data',
    # 500-class: the server's on-disk data for the requested date is incomplete
    # (a missing/duplicated variable COG), not something the client can fix by
    # changing the request. The detail names the date/variables (no paths).
    status=500,
)
