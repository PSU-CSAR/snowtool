"""Cross-cutting scalar types shared across snowdb, the CLI, and the API.

Just the genuine, behaviour-free type aliases: the :data:`StationTriplet`
constrained string (the workhorse, used wherever a pourpoint is identified). The
temporal *query objects* (with their ``select``/``csv_name`` behaviour) live in
:mod:`snowtool.snowdb.query`, and the triplet <-> filename-stem codec in
:mod:`snowtool.snowdb.triplet_naming` -- neither is a type.
"""

from typing import Annotated

from pydantic import Field, WithJsonSchema

STATION_TRIPLET = r'[a-zA-Z0-9\-]+:[a-zA-Z]{2}:[a-zA-Z]+'


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
