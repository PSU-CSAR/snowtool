"""Root router: the landing page + conformance + version/problems endpoints.

A :class:`~gazebo.ext.fastapi.RootRouter` auto-mounts ``/`` and ``/conformance``,
derived from the running app. Title/description fall back to the app's (set once
in ``get_app``). The cross-resource links to ``/datasets``/``/pourpoints`` are
declared here via :meth:`LinkedRouter.add_link` at import time, so re-running
``get_app`` (as tests do) never duplicates them.
"""

from __future__ import annotations

from gazebo.ext.fastapi import RootRouter
from gazebo.problems import ProblemType
from gazebo.rels import Rel

from snowtool.api.models.root import VersionInfo
from snowtool.api.problems import PROBLEM_TYPE_NOT_FOUND, PROBLEMS
from snowtool.api.tags import Tags

API_TITLE = 'PSU CSAR snowtool API'
API_DESCRIPTION = 'Pourpoint metadata and snow zonal statistics.'

# Router-level default tag so the auto-mounted `/` and `/conformance` routes are
# tagged too (they carry none of their own).
router: RootRouter = RootRouter(tags=[Tags.ROOT])
router.add_link(Rel.DATA, 'list_datasets', title='Datasets')
router.add_link(Rel.DATA, 'list_pourpoints', title='Pourpoints')


@router.get('/version', name='get_version')
async def get_version() -> VersionInfo:
    return VersionInfo()


@router.get('/problems', name='list_problems')
async def list_problems() -> dict[str, ProblemType]:
    """The catalog of problem types this API raises, keyed by short name.

    Each problem ``type`` URI resolves to its entry here, so a client can look up
    the meaning of a received ``application/problem+json`` ``type``.
    """
    return PROBLEMS.catalog()


@router.get('/problems/{key}', name='get_problem')
async def get_problem(key: str) -> ProblemType:
    problem = PROBLEMS.get(key)
    if problem is None:
        raise PROBLEM_TYPE_NOT_FOUND.exception(
            detail=f'No such problem type: {key!r}',
        )
    return problem
