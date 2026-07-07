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
            zones=[_zone_info(key, registry[key]) for key in sorted(registry)],
            links=[Link.self_link(), Link.root_link()],
        )


class DatasetList(BaseModel):
    datasets: list[str] = Field(examples=[['instarr', 'snodas', 'swann-800m']])
    links: list[Link] = Field(default_factory=list)

    @classmethod
    def from_snowdb(cls, snowdb: SnowDb) -> Self:
        return cls(
            datasets=sorted(snowdb),
            links=[
                Link.self_link(),
                Link.root_link(),
                *(
                    Link.to_route(
                        'get_dataset',
                        rel=Rel.ITEM,
                        title=name,
                        path={'dataset': name},
                    )
                    for name in sorted(snowdb)
                ),
            ],
        )
