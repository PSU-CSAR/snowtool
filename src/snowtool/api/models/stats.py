"""The zonal-stats response envelope and the CSV streaming helper.

The JSON response is a single generic envelope (:class:`CompactStatsResponse`),
shared by every dataset: the pourpoint ref, the echoed query, HATEOAS links, and a
compact body (zone layers / variables / zones defined once, values positional) so
there are no per-dataset field names and one OpenAPI schema covers them all.

Content is negotiated (``?f=`` / ``Accept``): ``csv`` streams
:meth:`ZonalStats.iter_csv` via :func:`stats_csv_response` with a
``Content-Disposition`` filename from the query's ``csv_name``; the JSON envelope
carries an ``alternate`` link to the CSV.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.responses import StreamingResponse
from gazebo.link import Link
from gazebo.negotiation import FormatEnum, f_description
from gazebo.params import DatetimeQuery
from gazebo.rels import MediaType, Rel
from pydantic import BaseModel, Field

from snowtool import types
from snowtool.snowdb.query import DateRangeQuery, DOYFields, DOYQuery, PourPointQuery
from snowtool.snowdb.zonal_stat_models import CompactStats

if TYPE_CHECKING:
    from collections.abc import Sequence

    from snowtool.snowdb.zonal_stats import ZonalStats


class StatsFormat(FormatEnum):
    """The ``?f=`` keys the stats route serves, each carrying its media type."""

    json = 'json', 'application/json'
    csv = 'csv', 'text/csv'


_ZONE_DESC = (
    'Stratify by a zone layer (repeatable): LAYER or LAYER:PARAM=VALUE. See '
    'GET /datasets/{dataset} for the valid layer keys, override params, and '
    'variables. Default: whole basin.'
)
_VARIABLE_DESC = (
    'Variable to report (repeatable; default: all). Use a variable key from '
    'GET /datasets/{dataset}.'
)
_ALLOW_PARTIAL_DESC = (
    'Permit a basin only partially covered by the dataset grid (default false: a '
    'partially-covered basin is a 409). A wholly off-grid basin always 409s.'
)
_INCLUDE_EMPTY_DESC = (
    'Include crossed zones that no AOI pixel falls in (0 area, all values null). '
    'By default these empty combinations are dropped. No effect on a whole-basin '
    'query.'
)
_DATETIME_EXAMPLES = [
    '2018-01-01/2018-06-30',
    '2018-04-27',
    '2018-01-01/..',
    '../2018-06-30',
]


class StatsQueryBase(BaseModel):
    zone: list[str] = Field(default_factory=list, description=_ZONE_DESC)
    variable: list[str] = Field(default_factory=list, description=_VARIABLE_DESC)
    allow_partial: bool = Field(default=False, description=_ALLOW_PARTIAL_DESC)
    include_empty_zones: bool = Field(default=False, description=_INCLUDE_EMPTY_DESC)
    f: StatsFormat | None = Field(
        default=None,
        description=f_description(StatsFormat),
    )


class DateRangeStatsQuery(StatsQueryBase):
    datetime: DatetimeQuery = Field(
        default=None,
        examples=_DATETIME_EXAMPLES,
        json_schema_extra={'example': _DATETIME_EXAMPLES[0]},
    )


class DOYStatsQuery(StatsQueryBase, DOYFields):
    pass


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
    """Stream a :class:`ZonalStats` as a CSV attachment named ``filename``.

    ``stats.iter_csv`` validates eagerly (before its first yield), so a bad
    result object still errors here -- before the response starts -- rather
    than mid-stream after the 200 headers are already sent.
    """
    return StreamingResponse(
        stats.iter_csv(include_empty_zones=include_empty_zones),
        media_type='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )
