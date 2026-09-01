from __future__ import annotations

from typing import Annotated

from fastapi import Query
from gazebo.ext.fastapi import GazeboRouter

from snowtool import types
from snowtool.api.dependencies import ReaderDep
from snowtool.api.models.snowline import SnowLineQuery, SnowLineResponse
from snowtool.api.tags import Tags
from snowtool.exceptions import QueryParameterError
from snowtool.snowdb.query import DateRangeQuery
from snowtool.snowdb.snowline import snow_line_elevation

router: GazeboRouter = GazeboRouter(prefix='/datasets/{dataset}/snowline')


@router.get(
    '/{triplet}/date-range',
    name='snowline_date_range',
    response_model=SnowLineResponse,
    tags=[Tags.SNOWLINE],
)
async def snowline_date_range(
    dataset: str,
    triplet: types.StationTriplet,
    reader: ReaderDep,
    params: Annotated[SnowLineQuery, Query()],
) -> SnowLineResponse:
    query = DateRangeQuery.from_interval(params.datetime)

    ds = reader.db.registered_dataset(dataset)
    try:
        unit = ds.spec.variables[params.variable].unit.name
    except KeyError:
        raise QueryParameterError(
            f'Variable {params.variable!r} is not in dataset {dataset!r}: '
            f'Available Stats: {ds.spec.variables.keys()}',
        ) from None

    stats = await reader.zonal_stats(
        triplet=triplet,
        dataset_name=dataset,
        query=query,
        variable_keys=(params.variable,),
        zones=(f'terrain.elevation:band_step_ft={params.band_step_ft}',),
        allow_partial=params.allow_partial,
    )

    results = snow_line_elevation(
        stats.dump_compact(include_empty_zones=False),
        threshold=params.threshold,
    )

    return SnowLineResponse.build(
        triplet=triplet,
        dataset=dataset,
        query=query,
        threshold=params.threshold,
        threshold_unit=unit,
        band_step_ft=params.band_step_ft,
        results=results,
    )
