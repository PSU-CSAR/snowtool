from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, assert_never

from gazebo.link import Link
from gazebo.rels import Rel
from pydantic import BaseModel, Field

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
    from snowtool.snowdb.zones.zone_layer import AvailableZone


class VariableInfo(BaseModel):
    """A requestable variable of a dataset and how its stat is reported."""

    key: str = Field(examples=['swe'])
    stat_name: str = Field(examples=['mean_swe_mm'])
    unit: str = Field(examples=['mm'])
    reducer: str = Field(examples=['mean'])


class ZoneClassInfo(BaseModel):
    """One categorical class a zone layer stratifies into (key + human label)."""

    key: str = Field(examples=['N'])
    label: str = Field(examples=['N'])


class BandedZoneInfo(BaseModel):
    """A banded zone axis: overridable band width over a covered numeric range.

    ``key`` is the value the ``zone`` query param accepts; the override query
    param is ``'<key>.<param>'`` (likewise the other overridable kinds below).
    """

    kind: Literal['banded'] = 'banded'
    key: str = Field(examples=['terrain.elevation'])
    param: str = Field(examples=['band_step_ft'])
    default: int = Field(examples=[1000])
    unit: str = Field(examples=['ft'])
    min: int | float = Field(examples=[-1000])
    max: int | float = Field(examples=[15000])


class BucketedZoneInfo(BaseModel):
    """An even-bucketed dimensionless axis: overridable bucket count."""

    kind: Literal['bucketed'] = 'bucketed'
    key: str = Field(examples=['terrain.northness'])
    param: str = Field(examples=['buckets'])
    default: int = Field(examples=[4])
    min: int | float = Field(examples=[-1])
    max: int | float = Field(examples=[1])


class ThresholdZoneInfo(BaseModel):
    """A threshold-split axis: overridable split point within its measured range."""

    kind: Literal['threshold'] = 'threshold'
    key: str = Field(examples=['landcover.forest_cover'])
    param: str = Field(examples=['threshold_pct'])
    default: float = Field(examples=[50.0])
    unit: str = Field(examples=['%'])
    min: int | float = Field(examples=[0])
    max: int | float = Field(examples=[100])


class CategoricalZoneInfo(BaseModel):
    """A categorical axis: fixed classes, no override param."""

    kind: Literal['categorical'] = 'categorical'
    key: str = Field(examples=['terrain.aspect'])
    classes: list[ZoneClassInfo]


# A stratifiable zone layer, discriminated on the scheme ``kind`` -- so the
# OpenAPI schema states the real per-kind contract (``classes`` exists iff
# categorical, ``param``/``default`` iff overridable) instead of advertising
# every field as nullable.
ZoneInfo = Annotated[
    BandedZoneInfo | BucketedZoneInfo | ThresholdZoneInfo | CategoricalZoneInfo,
    Field(discriminator='kind'),
]


def _zone_info(key: str, available: AvailableZone) -> ZoneInfo:
    """Build the per-kind :data:`ZoneInfo` member from a scheme's ``describe()``."""
    desc = available.scheme.describe()
    match desc:
        case BandedZoneDescription():
            return BandedZoneInfo(
                key=key,
                param=desc.param_key,
                default=desc.default,
                unit=desc.unit,
                min=desc.min,
                max=desc.max,
            )
        case BucketedZoneDescription():
            return BucketedZoneInfo(
                key=key,
                param=desc.param_key,
                default=desc.default,
                min=desc.min,
                max=desc.max,
            )
        case ThresholdZoneDescription():
            return ThresholdZoneInfo(
                key=key,
                param=desc.param_key,
                default=desc.default,
                unit=desc.unit,
                min=desc.min,
                max=desc.max,
            )
        case CategoricalZoneDescription():
            return CategoricalZoneInfo(
                key=key,
                classes=[ZoneClassInfo(key=c.key, label=c.label) for c in desc.classes],
            )
    assert_never(desc)


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

# Rels for the generic compact endpoint (Task 4): one route family across all
# datasets, so the query template is the fixed generic shape -- no per-zone
# override fields, no ``f`` negotiation (compact json is the only representation).
STATS_COMPACT_DATE_RANGE_REL = 'stats-compact-date-range'
STATS_COMPACT_DOY_REL = 'stats-compact-doy'


def dataset_zone_infos(dataset: Dataset) -> list[ZoneInfo]:
    """Every stratifiable zone of ``dataset`` as :data:`ZoneInfo`, sorted by key."""
    registry = available_zones(dataset.providers.values())
    return [_zone_info(key, registry[key]) for key in sorted(registry)]


def stats_links(
    name: str,
    zones: list[ZoneInfo],
    *,
    triplet: str | None = None,
) -> list[Link]:
    """Links to dataset ``name``'s two stats query endpoints.

    Without ``triplet`` -- the dataset resource's form -- the station triplet is
    an unbound RFC 6570 path variable (the dataset name is already baked into
    the route). With a ``triplet`` -- the pourpoint resource's form -- it is
    bound into the path, the titles are prefixed with the dataset name, and each
    link carries a machine-readable ``dataset`` field (a pass-through extra on
    gazebo's ``Link``), so a client holding several datasets' pairs selects one
    deterministically by ``(rel, dataset)`` instead of parsing titles or hrefs.
    Either way the query params are a form-query expansion: the shared
    zone/variable/negotiation params plus each overridable zone's
    ``<key>.<param>`` override field.
    """
    overrides = [
        f'{zone.key}.{zone.param}'
        for zone in zones
        if not isinstance(zone, CategoricalZoneInfo)
    ]
    shared = ['zone', 'variable', *overrides, 'allow_partial', 'f']
    if triplet is None:
        binding: dict[str, Any] = {'template': ['triplet']}
        date_range_title = 'Date-range zonal statistics'
        doy_title = 'Day-of-year zonal statistics'
    else:
        binding = {'path': {'triplet': triplet}, 'dataset': name}
        date_range_title = f'{name} date-range zonal statistics'
        doy_title = f'{name} day-of-year zonal statistics'
    compact_shared = ['zone', 'variable', 'allow_partial', 'include_empty_zones']
    if triplet is None:
        compact_binding: dict[str, Any] = {
            'path': {'dataset': name},
            'template': ['triplet'],
        }
    else:
        compact_binding = {
            'path': {'dataset': name, 'triplet': triplet},
            'dataset': name,
        }
    return [
        Link.to_route(
            f'{name}_stats_date_range',
            rel=STATS_DATE_RANGE_REL,
            title=date_range_title,
            query_template=['datetime', *shared],
            **binding,
        ),
        Link.to_route(
            f'{name}_stats_doy',
            rel=STATS_DOY_REL,
            title=doy_title,
            query_template=['month', 'day', 'start_year', 'end_year', *shared],
            **binding,
        ),
        Link.to_route(
            'stats_compact_date_range',
            rel=STATS_COMPACT_DATE_RANGE_REL,
            title=(
                'Compact date-range zonal statistics'
                if triplet is None
                else f'{name} compact date-range zonal statistics'
            ),
            query_template=['datetime', *compact_shared],
            **compact_binding,
        ),
        Link.to_route(
            'stats_compact_doy',
            rel=STATS_COMPACT_DOY_REL,
            title=(
                'Compact day-of-year zonal statistics'
                if triplet is None
                else f'{name} compact day-of-year zonal statistics'
            ),
            query_template=['month', 'day', 'start_year', 'end_year', *compact_shared],
            **compact_binding,
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
                *stats_links(spec.name, zones),
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
