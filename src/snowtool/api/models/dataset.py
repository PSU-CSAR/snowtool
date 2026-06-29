from __future__ import annotations

from typing import TYPE_CHECKING, Self

from gazebo.link import Link
from gazebo.rels import Rel
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.db import SnowDb


class VariableInfo(BaseModel):
    """A requestable variable of a dataset and how its stat is reported."""

    key: str
    stat_name: str
    unit: str
    reducer: str


class GridInfo(BaseModel):
    """A summary of a dataset's tiled grid."""

    crs: str
    rows: int
    cols: int
    tile_size: int
    is_geographic: bool


class DatasetInfo(BaseModel):
    name: str
    grid: GridInfo
    variables: list[VariableInfo]
    links: list[Link] = Field(default_factory=list)

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> Self:
        spec = dataset.spec
        grid_params = spec.grid_params
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
            links=[Link.self_link(), Link.root_link()],
        )


class DatasetList(BaseModel):
    datasets: list[str]
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
