from __future__ import annotations

from fastapi import APIRouter, Request

from snowtool.api.models.root import LandingPage, VersionInfo
from snowtool.api.tags import Tags

router: APIRouter = APIRouter()


@router.get(
    '/',
    tags=[Tags.ROOT],
)
async def get_landing_page(request: Request) -> LandingPage:
    return LandingPage.from_request(request)


@router.get(
    '/version',
    tags=[Tags.ROOT],
)
async def get_version(request: Request) -> VersionInfo:
    return VersionInfo.from_request(request)
