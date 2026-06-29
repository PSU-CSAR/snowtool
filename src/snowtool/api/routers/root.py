"""Root router: the landing page + version endpoint.

A :class:`~gazebo.ext.fastapi.LinkedRouter` auto-mounts the landing page at ``/``
(route name ``landing``) with ``self`` + ``root`` links and a ``gazebo.ogc``
``LandingPage`` response model. The cross-resource links to the collection
endpoints (``/datasets``, ``/aois``) are declared here via
:meth:`LinkedRouter.add_link`, which takes route *names* (resolved at request time
via ``url_for``) -- so the sibling routers need not be imported, and the calls run
once at import (not per ``get_app``, where the shared router would accumulate
duplicates).
"""

from __future__ import annotations

from gazebo.ext.fastapi import LinkedRouter
from gazebo.rels import Rel

from snowtool.api.models.root import VersionInfo
from snowtool.api.tags import Tags

API_TITLE = 'PSU CSAR snowtool API'
API_DESCRIPTION = 'Pourpoint/AOI metadata and snow zonal statistics.'

router: LinkedRouter = LinkedRouter(title=API_TITLE, description=API_DESCRIPTION)
router.add_link(Rel.DATA, 'list_datasets', title='Datasets')
router.add_link(Rel.DATA, 'list_aois', title='AOIs')


@router.get('/version', name='get_version', tags=[Tags.ROOT])
async def get_version() -> VersionInfo:
    return VersionInfo.build()
