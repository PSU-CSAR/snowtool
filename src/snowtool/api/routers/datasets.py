from __future__ import annotations

from fastapi import APIRouter, Request

from snowtool.api.exceptions import NotFoundError
from snowtool.api.models.dataset import DatasetInfo, DatasetList
from snowtool.api.tags import Tags

router: APIRouter = APIRouter()


@router.get(
    '/datasets',
    tags=[Tags.DATASETS],
)
async def list_datasets(request: Request) -> DatasetList:
    return DatasetList.from_snowdb(request.state.snowdb, request)


@router.get(
    '/datasets/{dataset}',
    tags=[Tags.DATASETS],
)
async def get_dataset(request: Request, dataset: str) -> DatasetInfo:
    snowdb = request.state.snowdb
    try:
        bound = snowdb[dataset]
    except KeyError as e:
        raise NotFoundError(f'No such dataset: {dataset!r}') from e
    return DatasetInfo.from_dataset(bound, request)
