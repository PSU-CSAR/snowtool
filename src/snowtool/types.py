"""Cross-cutting scalar types shared across snowdb, the CLI, and the API.

Just the genuine, behaviour-free type aliases and their validation primitives:
the :data:`StationTriplet` constrained string (the workhorse, used wherever a
pourpoint is identified) and :func:`to_date`. The temporal *query objects* (with
their ``select``/``csv_name`` behaviour) live in :mod:`snowtool.snowdb.query`, and
the triplet <-> filename-stem codec in :mod:`snowtool.snowdb.triplet_naming` --
neither is a type.
"""

from datetime import date, datetime
from typing import Annotated

from pydantic import Field, WithJsonSchema

STATION_TRIPLET = r'[a-zA-Z0-9\-]+:[a-zA-Z]{2}:[a-zA-Z]+'


def to_date(value: str) -> date:
    return datetime.strptime(  # noqa: DTZ007
        value.replace('-', ''),
        '%Y%m%d',
    ).date()


StationTriplet = Annotated[
    str,
    Field(
        pattern=f'^{STATION_TRIPLET}$',
    ),
    WithJsonSchema(
        {
            'example': '12354500:MT:USGS',
        },
        mode='validation',
    ),
]
