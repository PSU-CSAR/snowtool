from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Self

from gazebo.link import Link
from gazebo.rels import Rel
from pydantic import BaseModel, Field, TypeAdapter

from snowtool.snowdb.zones.zone_layer import available_zones
from snowtool.snowdb.zones.zoning import (
    BandedZoneDescription,
    BucketedZoneDescription,
    CategoricalZoneDescription,
    ThresholdZoneDescription,
)

if TYPE_CHECKING:
    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.db import SnowDb


class VariableInfo(BaseModel):
    """A requestable variable of a dataset and how its stat is reported."""

    key: str = Field(examples=['swe'])
    stat_name: str = Field(examples=['mean_swe_mm'])
    unit: str = Field(examples=['mm'])
    reducer: str = Field(examples=['mean'])


class BandedZoneInfo(BandedZoneDescription):
    """A banded zone axis: overridable band width over a covered numeric range.

    ``key`` is the value the ``zone`` query param accepts; the override query
    param is ``'<key>.<param>'`` (likewise the other overridable kinds below).
    """

    key: str = Field(examples=['terrain.elevation'])


class BucketedZoneInfo(BucketedZoneDescription):
    """An even-bucketed dimensionless axis: overridable bucket count."""

    key: str = Field(examples=['terrain.northness'])


class ThresholdZoneInfo(ThresholdZoneDescription):
    """A threshold-split axis: overridable split point within its measured range."""

    key: str = Field(examples=['landcover.forest_cover'])


class CategoricalZoneInfo(CategoricalZoneDescription):
    """A categorical axis: fixed classes, no override param."""

    key: str = Field(examples=['terrain.aspect'])


# A stratifiable zone layer, discriminated on the scheme ``kind`` -- so the
# OpenAPI schema states the real per-kind contract (``classes`` exists iff
# categorical, ``param``/``default`` iff overridable) instead of advertising
# every field as nullable. Each member is the domain description plus a registry
# ``key``, so the wire schema is served straight from the description.
ZoneInfo = Annotated[
    BandedZoneInfo | BucketedZoneInfo | ThresholdZoneInfo | CategoricalZoneInfo,
    Field(discriminator='kind'),
]

_zone_info_adapter: TypeAdapter[ZoneInfo] = TypeAdapter(ZoneInfo)


class GridInfo(BaseModel):
    """A summary of a dataset's tiled grid."""

    crs: str = Field(examples=['4326'])
    rows: int = Field(examples=[3351])
    cols: int = Field(examples=[6935])
    tile_size: int = Field(examples=[256])
    is_geographic: bool = Field(examples=[True])


# Custom link relations for a dataset's two queryable stats endpoints. No IANA
# relation fits a parameterized query sub-resource, so these are service-specific
# tokens; the templated links they tag leave the station triplet and query params
# for the client to bind.
STATS_DATE_RANGE_REL = 'stats-date-range'
STATS_DOY_REL = 'stats-doy'


def dataset_zone_infos(dataset: Dataset) -> list[ZoneInfo]:
    """Every stratifiable zone of ``dataset`` as :data:`ZoneInfo`, sorted by key.

    Each info is the scheme's own ``describe()`` description plus the registry
    ``key``; the ``kind`` field discriminates the right :data:`ZoneInfo` member.
    """
    registry = available_zones(dataset.providers.values())
    return [
        _zone_info_adapter.validate_python(
            {'key': key, **dict(registry[key].scheme.describe())},
        )
        for key in sorted(registry)
    ]


_SHARED_STATS_QUERY = ['zone', 'variable', 'allow_partial', 'include_empty_zones', 'f']
_DOY_QUERY = ['month', 'day', 'start_year', 'end_year', *_SHARED_STATS_QUERY]


def dataset_stats_links(name: str) -> list[Link]:
    """Links to dataset ``name``'s two stats query endpoints, the dataset resource's
    form: the station triplet is an unbound RFC 6570 path variable, and titles are
    unprefixed. The query params are the generic form-query expansion (no per-zone
    override fields; ``f`` negotiates json/csv).
    """
    return [
        Link.to_route(
            'stats_date_range',
            rel=STATS_DATE_RANGE_REL,
            title='Date-range zonal statistics',
            path={'dataset': name},
            template=['triplet'],
            query_template=['datetime', *_SHARED_STATS_QUERY],
        ),
        Link.to_route(
            'stats_doy',
            rel=STATS_DOY_REL,
            title='Day-of-year zonal statistics',
            path={'dataset': name},
            template=['triplet'],
            query_template=_DOY_QUERY,
        ),
    ]


def pourpoint_stats_links(name: str, triplet: str) -> list[Link]:
    """Links to dataset ``name``'s two stats query endpoints, the pourpoint
    resource's form: the station triplet is bound into the path, titles are
    dataset-prefixed, and each link carries a machine-readable ``dataset`` field
    so a client holding several datasets' pairs selects one by ``(rel, dataset)``.
    The query params are the generic form-query expansion (no per-zone override
    fields; ``f`` negotiates json/csv).
    """
    return [
        Link.to_route(
            'stats_date_range',
            rel=STATS_DATE_RANGE_REL,
            title=f'{name} date-range zonal statistics',
            path={'dataset': name, 'triplet': triplet},
            dataset=name,
            query_template=['datetime', *_SHARED_STATS_QUERY],
        ),
        Link.to_route(
            'stats_doy',
            rel=STATS_DOY_REL,
            title=f'{name} day-of-year zonal statistics',
            path={'dataset': name, 'triplet': triplet},
            dataset=name,
            query_template=_DOY_QUERY,
        ),
    ]


class DatasetInfo(BaseModel):
    name: str = Field(examples=['snodas'])
    grid: GridInfo
    variables: list[VariableInfo]
    zones: list[ZoneInfo]
    links: list[Link] = Field(default_factory=list)

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> Self:
        spec = dataset.spec
        grid_params = spec.grid_params
        zones = dataset_zone_infos(dataset)
        return cls(
            name=spec.name,
            grid=GridInfo(
                crs=str(grid_params.crs),
                rows=grid_params.rows,
                cols=grid_params.cols,
                tile_size=grid_params.tile_size,
                is_geographic=spec.is_geographic,
            ),
            variables=[
                VariableInfo(
                    key=variable.key,
                    stat_name=variable.stat_name,
                    unit=variable.unit.name,
                    reducer=str(variable.reducer),
                )
                for variable in spec.variables.values()
            ],
            zones=zones,
            links=[
                Link.self_link(),
                Link.root_link(),
                *dataset_stats_links(spec.name),
            ],
        )


class DatasetListItem(BaseModel):
    """One entry in the dataset list: its name plus a ``self`` link to its detail
    route, so each item is a followable resource rather than a bare string paired
    with a link stranded in the collection's ``links`` array."""

    name: str = Field(examples=['snodas'])
    links: list[Link] = Field(default_factory=list)


class DatasetList(BaseModel):
    datasets: list[DatasetListItem]
    links: list[Link] = Field(default_factory=list)

    @classmethod
    def from_snowdb(cls, snowdb: SnowDb) -> Self:
        return cls(
            datasets=[
                DatasetListItem(
                    name=name,
                    links=[
                        Link.to_route(
                            'get_dataset',
                            rel=Rel.SELF,
                            path={'dataset': name},
                        ),
                    ],
                )
                for name in sorted(snowdb)
            ],
            links=[Link.self_link(), Link.root_link()],
        )
