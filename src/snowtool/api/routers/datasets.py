"""Dataset catalog routes (list + item lookup).

:func:`build_datasets_router` is called once in ``get_app`` so the ``{dataset}``
path param can advertise the catalog's dataset names. The valid set is small and
fixed at app-build time (exactly ``catalog.datasets``, the same list the stats
routers loop over), so we surface it as an OpenAPI ``enum`` on the param -- a
dropdown in the docs and a closed set for client codegen.

The param stays typed ``str`` on purpose: the ``enum`` rides in via
``json_schema_extra`` (schema metadata only, *not* a validation constraint), so an
unknown dataset still flows into the handler and 404s through the
``dataset-not-found`` problem response -- the RESTful "no such resource" answer --
rather than a schema-layer 422. (Contrast ``zone`` in the stats router: a *query*
param, where an enum constraint and its 422 are the right call.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Path
from gazebo.ext.fastapi import GazeboRouter

from snowtool.api.dependencies import CatalogDb
from snowtool.api.models.dataset import DatasetInfo, DatasetList
from snowtool.api.problems import DATASET_NOT_FOUND
from snowtool.api.tags import Tags

if TYPE_CHECKING:
    from snowtool.snowdb.db import SnowDb


def build_datasets_router(catalog: SnowDb) -> GazeboRouter:
    """A :class:`GazeboRouter` with the catalog list + item-lookup routes, the
    latter advertising the catalog's dataset names as an OpenAPI ``enum``."""
    router: GazeboRouter = GazeboRouter()
    dataset_param = Annotated[
        str,
        Path(
            description='Dataset name.',
            json_schema_extra={'enum': sorted(catalog)},
        ),
    ]

    @router.get('/datasets', name='list_datasets', tags=[Tags.DATASETS])
    async def list_datasets(snowdb: CatalogDb) -> DatasetList:
        return DatasetList.from_snowdb(snowdb)

    async def get_dataset(dataset, snowdb: CatalogDb) -> DatasetInfo:
        try:
            bound = snowdb[dataset]
        except KeyError as e:
            raise DATASET_NOT_FOUND.exception(
                detail=f'No such dataset: {dataset!r}',
            ) from e
        return DatasetInfo.from_dataset(bound)

    # ``dataset_param`` is a local (built per-catalog), so it cannot be named in the
    # annotation: ``from __future__ import annotations`` would stringify the hint and
    # neither FastAPI nor gazebo could resolve the local name. Patch the *real*
    # Annotated object onto ``__annotations__`` before the route is registered (which
    # is when both introspect the signature). Mirrors ``stats.build_stats_router``.
    get_dataset.__annotations__['dataset'] = dataset_param
    router.get('/datasets/{dataset}', name='get_dataset', tags=[Tags.DATASETS])(
        get_dataset,
    )

    return router
