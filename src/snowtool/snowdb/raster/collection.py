from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from typing import TYPE_CHECKING, Self

from snowtool.snowdb.raster.tiled import DataRaster

if TYPE_CHECKING:
    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.query import DateQuery
    from snowtool.snowdb.variables import DatasetVariable


class RasterCollection:
    def __init__(
        self: Self,
        query: DateQuery,
        rasters: dict[DatasetVariable, list[DataRaster]],
    ) -> None:
        self.query = query
        self._by_variable = {
            variable: sorted(rs, key=lambda x: x.date)
            for variable, rs in rasters.items()
        }
        self._by_date: dict[date, set[DatasetVariable]] = {}
        for variable, rs in self._by_variable.items():
            for raster in rs:
                self._by_date.setdefault(raster.date, set()).add(variable)

        self.validate()

    @property
    def variables(self: Self) -> set[DatasetVariable]:
        return set(self._by_variable.keys())

    @property
    def dates(self: Self) -> list[date]:
        return sorted(self._by_date.keys())

    def items(self: Self) -> Iterator[tuple[DatasetVariable, list[DataRaster]]]:
        yield from self._by_variable.items()

    def __iter__(self: Self) -> Iterator[DataRaster]:
        for rasters in self._by_variable.values():
            yield from rasters

    def __len__(self: Self) -> int:
        return sum(len(rasters) for rasters in self._by_variable.values())

    @classmethod
    def from_variables_query(
        cls: type[Self],
        query: DateQuery,
        variables: set[DatasetVariable],
        dataset: Dataset,
    ) -> Self:
        return cls(
            query=query,
            rasters={
                variable: [
                    DataRaster(path, date_)
                    for date_, path in dataset.raster_paths_from_query(
                        query,
                        variable,
                    )
                ]
                for variable in variables
            },
        )

    def validate(self: Self) -> None:
        if len({len(rasters) for rasters in self._by_variable.values()}) > 1:
            raise ValueError('Variable raster lists are not all the same length')

        expected = self.variables
        for date_, present in self._by_date.items():
            if present != expected:
                raise ValueError(
                    f"Unexpected variable set for date '{date_}': "
                    f'{sorted(v.key for v in expected)} != '
                    f'{sorted(v.key for v in present)}',
                )
