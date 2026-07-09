from __future__ import annotations

from typing import TYPE_CHECKING, Self

from gazebo.link import Link
from gazebo.rels import Rel
from pydantic import BaseModel, Field

from snowtool.snowdb.zones.zone_layer import available_zones

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


class ZoneInfo(BaseModel):
    """A stratifiable zone layer of a dataset and how to override its scheme.

    ``key`` is the value the ``zone`` query param accepts; ``kind`` is the scheme
    kind. An overridable layer advertises its ``param`` (the override query param
    is ``'<key>.<param>'``), the scheme ``default`` for it, and the ``unit``; a
    categorical layer instead advertises its ``classes`` (and has no ``param``).
    """

    key: str = Field(examples=['terrain.elevation'])
    kind: str = Field(examples=['banded'])
    param: str | None = Field(default=None, examples=['band_step_ft'])
    default: float | int | None = Field(default=None, examples=[1000])
    unit: str | None = Field(default=None, examples=['ft'])
    classes: list[ZoneClassInfo] | None = Field(default=None)


def _zone_info(key: str, available: AvailableZone) -> ZoneInfo:
    """Build a :class:`ZoneInfo` from a registry entry's scheme ``describe()``."""
    desc = available.scheme.describe()
    return ZoneInfo(
        key=key,
        kind=desc.kind,
        param=desc.param_key,
        default=desc.default,
        unit=desc.unit,
        classes=(
            [ZoneClassInfo(key=c.key, label=c.label) for c in desc.classes]
            if desc.classes is not None
            else None
        ),
    )


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


def _stats_links(name: str, zones: list[ZoneInfo]) -> list[Link]:
    """Templated links to dataset ``name``'s two stats query endpoints.

    The station triplet is an unbound RFC 6570 path variable (the dataset name is
    already baked into the route) and the query params are a form-query expansion --
    the shared zone/variable/negotiation params plus each overridable zone's
    ``<key>.<param>`` override field -- so a client can build a stats query from the
    dataset resource alone.
    """
    overrides = [f'{zone.key}.{zone.param}' for zone in zones if zone.param]
    shared = ['zone', 'variable', *overrides, 'allow_partial', 'f']
    return [
        Link.to_route(
            f'{name}_stats_date_range',
            rel=STATS_DATE_RANGE_REL,
            title='Date-range zonal statistics',
            template=['triplet'],
            query_template=['datetime', *shared],
        ),
        Link.to_route(
            f'{name}_stats_doy',
            rel=STATS_DOY_REL,
            title='Day-of-year zonal statistics',
            template=['triplet'],
            query_template=['month', 'day', 'start_year', 'end_year', *shared],
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
        registry = available_zones(dataset.providers.values())
        zones = [_zone_info(key, registry[key]) for key in sorted(registry)]
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
                *_stats_links(spec.name, zones),
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
