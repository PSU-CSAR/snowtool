from __future__ import annotations

from gazebo.ext.fastapi import GazeboRouter

from snowtool.api.dependencies import CatalogDb
from snowtool.api.models.dataset import DatasetInfo, DatasetList
from snowtool.api.tags import Tags

router: GazeboRouter = GazeboRouter()


@router.get('/datasets', name='list_datasets', tags=[Tags.DATASETS])
async def list_datasets(snowdb: CatalogDb) -> DatasetList:
    return DatasetList.from_snowdb(snowdb)


@router.get('/datasets/{dataset}', name='get_dataset', tags=[Tags.DATASETS])
async def get_dataset(dataset: str, snowdb: CatalogDb) -> DatasetInfo:
    bound = snowdb[dataset]
    return DatasetInfo.from_dataset(bound)
