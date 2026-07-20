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

from typing import TYPE_CHECKING

from fastapi.responses import StreamingResponse
from gazebo.link import Link
from gazebo.rels import MediaType, Rel
from pydantic import Field

from snowtool import types
from snowtool.snowdb.query import DateRangeQuery, DOYQuery, PourPointQuery
from snowtool.snowdb.zonal_stat_models import CompactStats

if TYPE_CHECKING:
    from collections.abc import Sequence

    from snowtool.snowdb.zonal_stats import ZonalStats


class CompactStatsResponse(CompactStats):
    """The compact zonal-stats envelope: pourpoint/query echo + the compact body.

    Inherits the body fields (zone_layers / variables / zones / results) from
    :class:`CompactStats` and adds the HTTP-only envelope: the pourpoint/query echo
    and HATEOAS links. Inheriting rather than re-declaring the body keeps the
    response in lockstep with the domain body -- a new stat field flows through
    automatically -- and keeps the ``Link``/route concerns out of the domain model.
    A single generic model shared by every dataset (the body carries no per-dataset
    field names), so it is not parametrized per dataset.
    """

    pourpoint: types.StationTriplet
    dataset: str = Field(examples=['snodas'])
    query: PourPointQuery
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
            # Splat the domain body straight in: no field is enumerated here, so a
            # new CompactStats field needs no change to this envelope.
            **dict(stats),
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
