"""The zonal-stats response envelope and the CSV streaming helper.

The JSON response is a thin per-dataset envelope -- the pourpoint ref, the echoed query,
HATEOAS links, and ``results``: a list of the dataset's *generated* per-date model
(:attr:`DatasetSpec.zonal_stats_model`). Parametrizing the generic
:class:`StatsResponse` with that model (``StatsResponse[spec.zonal_stats_model]``)
gives each dataset a precise OpenAPI schema, which is exactly what the
``model_prefix`` uniqueness check in ``db._index_specs`` protects.

Content is negotiated (``?f=`` / ``Accept``): ``csv`` streams
:meth:`ZonalStats.dump_to_csv` with a ``Content-Disposition`` filename from the
query's ``csv_name``; the JSON envelope carries an ``alternate`` link to the CSV.
"""

from __future__ import annotations

import io

from typing import TYPE_CHECKING

from fastapi.responses import StreamingResponse
from gazebo.link import Link
from gazebo.rels import MediaType, Rel
from pydantic import BaseModel, Field

from snowtool import types
from snowtool.snowdb.query import DateRangeQuery, DOYQuery, PourPointQuery

if TYPE_CHECKING:
    from collections.abc import Sequence

    from snowtool.snowdb.zonal_stats import ZonalStats


class StatsResponse[T](BaseModel):
    """A per-dataset zonal-stats envelope: pourpoint/query echo + results + links."""

    pourpoint: types.StationTriplet
    dataset: str
    query: PourPointQuery
    results: list[T]
    links: list[Link] = Field(default_factory=list)

    @classmethod
    def build(
        cls,
        *,
        triplet: types.StationTriplet,
        dataset: str,
        query: DateRangeQuery | DOYQuery,
        results: list[T],
        alternates: Sequence[Link] = (),
    ) -> StatsResponse[T]:
        return cls(
            pourpoint=triplet,
            dataset=dataset,
            query=query,
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


def stats_csv_response(stats: ZonalStats, filename: str) -> StreamingResponse:
    """Stream a :class:`ZonalStats` as a CSV attachment named ``filename``."""
    buffer = io.StringIO()
    stats.dump_to_csv(buffer)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )
