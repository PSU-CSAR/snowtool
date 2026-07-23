"""Cross-cutting scalar types shared across snowdb, the CLI, and the API.

Just the genuine, behaviour-free type aliases: the :data:`StationTriplet`
constrained string (the workhorse, used wherever a pourpoint is identified). The
temporal *query objects* (with their ``select``/``date_fragment`` behaviour) live in
:mod:`snowtool.snowdb.query`, and the triplet <-> filename-stem codec in
:mod:`snowtool.snowdb.triplet_naming` -- neither is a type.
"""

from typing import Annotated

from pydantic import Field

STATION_TRIPLET = r'[a-zA-Z0-9\-]+:[a-zA-Z]{2}:[a-zA-Z]+'


# ``Field(examples=...)`` rather than ``WithJsonSchema(..., mode='validation')``:
# the latter applies only to validation-mode schemas, so the example would be
# invisible in every *response* schema (FastAPI renders those in serialization
# mode) and Swagger would show a regex-generated random triplet instead.
StationTriplet = Annotated[
    str,
    Field(
        pattern=f'^{STATION_TRIPLET}$',
        examples=['12354500:MT:USGS'],
    ),
]
