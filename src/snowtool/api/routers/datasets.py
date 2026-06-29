from __future__ import annotations

from typing import Annotated

from gazebo.ext.fastapi import GazeboRouter, Inject
from gazebo.problems import ProblemException

from snowtool.api.models.dataset import DatasetInfo, DatasetList
from snowtool.api.tags import Tags
from snowtool.snowdb.db import SnowDb

# SnowDb is an app-scoped constant provider (no __provide__), so injection is
# opt-in via the Inject marker.
CatalogDb = Annotated[SnowDb, Inject]

router: GazeboRouter = GazeboRouter()


@router.get('/datasets', name='list_datasets', tags=[Tags.DATASETS])
async def list_datasets(snowdb: CatalogDb) -> DatasetList:
    return DatasetList.from_snowdb(snowdb)


@router.get('/datasets/{dataset}', name='get_dataset', tags=[Tags.DATASETS])
async def get_dataset(dataset: str, snowdb: CatalogDb) -> DatasetInfo:
    try:
        bound = snowdb[dataset]
    except KeyError as e:
        raise ProblemException(404, detail=f'No such dataset: {dataset!r}') from e
    return DatasetInfo.from_dataset(bound)
