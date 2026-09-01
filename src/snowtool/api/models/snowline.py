"""snowline response model, effectively parroted from CompactStatsResponse
in stats.py

"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from gazebo.link import Link
from gazebo.params import DatetimeQuery
from gazebo.rels import MediaType, Rel
from pydantic import BaseModel, Field

from snowtool import types
from snowtool.snowdb.query import DateRangeQuery, DOYQuery, PourPointQuery
from snowtool.snowdb.snowline import SnowLine

if TYPE_CHECKING:
    from collections.abc import Sequence


class SnowLineQuery(BaseModel):
    datetime: DatetimeQuery = Field(default=None, examples=['2026-04-01/2026-07-01'])
    variable: str = Field(
        default='snow_fraction',
        description='Variable to be used to calculate snowline',
    )
    threshold: float = Field(
        default=50.0,
        gt=0.0,
        description='Value at which snow line is drawn.',
    )
    band_step_ft: int = Field(
        default=500,
        gt=0,
        description='Elevation band step. Smaller gives finer resolution but '
        'fewer pixels per band.',
    )
    allow_partial: bool = Field(default=False)


class SnowLineResponse(BaseModel):
    pourpoint: types.StationTriplet
    dataset: str = Field(examples=['instarr'])
    query: PourPointQuery
    threshold: float
    threshold_unit: str = Field(examples=['mm,percent'])
    band_step_ft: int
    results: dict[date, SnowLine | None]
    links: list[Link] = Field(default_factory=list)

    @classmethod
    def build(
        cls,
        *,
        triplet: types.StationTriplet,
        dataset: str,
        query: DateRangeQuery | DOYQuery,
        threshold: float,
        threshold_unit: str,
        band_step_ft: int,
        results: dict[date, SnowLine | None],
        alternates: Sequence[Link] = (),
    ) -> SnowLineResponse:
        return cls(
            pourpoint=triplet,
            dataset=dataset,
            query=query,
            threshold=threshold,
            threshold_unit=threshold_unit,
            band_step_ft=band_step_ft,
            results=results,
            links=[
                Link.self_link(),
                Link.root_link(),
                Link.to_route(
                    'get_pourpoint',
                    rel=Rel.UP,
                    type=MediaType.GEOJSON,
                    path={'triplet': triplet},
                ),
                *alternates,
            ],
        )
