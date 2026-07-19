"""The zonal-stats response envelope and the CSV streaming helper.

The JSON response is a single generic envelope (:class:`CompactStatsResponse`),
shared by every dataset: the pourpoint ref, the echoed query, HATEOAS links, and a
compact body (zone layers / variables / zones defined once, values positional) so
there are no per-dataset field names and one OpenAPI schema covers them all.

Content is negotiated (``?f=`` / ``Accept``): ``csv`` streams
:meth:`ZonalStats.dump_to_csv` via :func:`stats_csv_response` with a
``Content-Disposition`` filename from the query's ``csv_name``; the JSON envelope
carries an ``alternate`` link to the CSV.
"""

from __future__ import annotations

import io

from datetime import date
from typing import TYPE_CHECKING

from fastapi.responses import StreamingResponse
from gazebo.link import Link
from gazebo.rels import MediaType, Rel
from pydantic import BaseModel, Field

from snowtool import types
from snowtool.snowdb.query import DateRangeQuery, DOYQuery, PourPointQuery
from snowtool.snowdb.zonal_stat_models import CompactStats, CompactZone, StatValue

if TYPE_CHECKING:
    from collections.abc import Sequence

    from snowtool.snowdb.zonal_stats import ZonalStats


class CompactStatsResponse(BaseModel):
    """The compact zonal-stats envelope: pourpoint/query echo + the compact body.

    A single generic model shared by every dataset — the compact body carries no
    per-dataset field names — so it is not parametrized per dataset.
    """

    pourpoint: types.StationTriplet
    dataset: str = Field(examples=['snodas'])
    query: PourPointQuery
    zone_layers: list[str] = Field(examples=[['terrain.elevation']])
    variables: list[str] = Field(examples=[['mean_swe_mm']])
    zones: list[CompactZone]
    results: dict[date, list[list[StatValue]]] = Field(
        examples=[{'2008-12-14': [[42.7], [51.3]]}],
    )
    links: list[Link] = Field(default_factory=list)

    @classmethod
    def build(
        cls,
        *,
        triplet: types.StationTriplet,
        dataset: str,
        query: DateRangeQuery | DOYQuery,
        stats: CompactStats,
        alternates: Sequence[Link] = (),
    ) -> CompactStatsResponse:
        return cls(
            pourpoint=triplet,
            dataset=dataset,
            query=query,
            zone_layers=stats.zone_layers,
            variables=stats.variables,
            zones=stats.zones,
            results=stats.results,
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


def stats_csv_response(
    stats: ZonalStats,
    filename: str,
    *,
    include_empty_zones: bool = False,
) -> StreamingResponse:
    """Stream a :class:`ZonalStats` as a CSV attachment named ``filename``."""
    buffer = io.StringIO()
    stats.dump_to_csv(buffer, include_empty_zones=include_empty_zones)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )
