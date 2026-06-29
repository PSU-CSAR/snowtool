"""Root router: the landing page + conformance + version/problems endpoints.

A :class:`~gazebo.ext.fastapi.RootRouter` (the service-root ``LinkedRouter``)
auto-mounts the landing page at ``/`` (route name ``landing``) with ``self`` +
``root`` links and a ``gazebo.ogc`` ``LandingPage`` response model, and -- derived
from the *running app* so it can't drift -- a ``/conformance`` declaration plus
``service-desc``/``service-doc`` links to the OpenAPI doc and docs UI. Title and
description fall back to the app's (set once in ``get_app``), so they live in one
place. The cross-resource links to the collection endpoints (``/datasets``,
``/pourpoints``) are declared here via :meth:`LinkedRouter.add_link`, which takes
route *names* (resolved at request time via ``url_for``) -- so the sibling routers
need not be imported, and the calls run once at import (not per ``get_app``, where
the shared router would accumulate duplicates).
"""

from __future__ import annotations

from gazebo.ext.fastapi import RootRouter
from gazebo.problems import ProblemType
from gazebo.rels import Rel

from snowtool.api.models.root import VersionInfo
from snowtool.api.problems import PROBLEMS
from snowtool.api.tags import Tags

API_TITLE = 'PSU CSAR snowtool API'
API_DESCRIPTION = 'Pourpoint metadata and snow zonal statistics.'

router: RootRouter = RootRouter()
router.add_link(Rel.DATA, 'list_datasets', title='Datasets')
router.add_link(Rel.DATA, 'list_pourpoints', title='Pourpoints')


@router.get('/version', name='get_version', tags=[Tags.ROOT])
async def get_version() -> VersionInfo:
    return VersionInfo.build()


@router.get('/problems', name='list_problems', tags=[Tags.ROOT])
async def list_problems() -> dict[str, ProblemType]:
    """The catalog of problem types this API raises, keyed by short name.

    Each problem ``type`` URI resolves to its entry here, so a client can look up
    the meaning of a received ``application/problem+json`` ``type``.
    """
    return PROBLEMS.catalog()


@router.get('/problems/{key}', name='get_problem', tags=[Tags.ROOT])
async def get_problem(key: str) -> ProblemType:
    problem = PROBLEMS.get(key)
    if problem is None:
        raise PROBLEMS['problem-type-not-found'].exception(
            detail=f'No such problem type: {key!r}',
        )
    return problem
