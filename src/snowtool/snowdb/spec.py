"""Dataset specifications: the path-independent *definition* of a dataset kind.

A :class:`DatasetSpec` is the "SNODAS-ness" of a dataset — its grid, DEM
elevation range, and (later phases) its variables and ingest. The built-in specs
live in :mod:`snowtool.snowdb.datasets` and are passed in to a
:class:`~snowtool.snowdb.db.SnowDb`; a spec exists with or without data on disk.
A :class:`Dataset` (see :mod:`snowtool.snowdb.dataset`) binds a spec to a
``data/<name>/`` path.

The spec carries behavior (it builds its grid, derives ``cell_area``) because it
is a definition, not a passive settings bag.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from pyproj import CRS

from snowtool.snowdb.grid import make_grid

if TYPE_CHECKING:
    from collections.abc import Iterable

    from snowtool.snowdb.variables import DatasetVariable


@dataclass(frozen=True)
class GridParams:
    """The parameters defining a dataset's north-up tiled grid."""

    origin_x: float
    origin_y: float
    px_size: float
    cols: int
    rows: int
    tile_size: int
    crs: int | str = 4326


class DatasetSpec:
    def __init__(
        self,
        name: str,
        *,
        grid_params: GridParams,
        dem_min_m: float,
        dem_max_m: float,
        variables: Iterable[DatasetVariable] = (),
        band_step_ft: int = 1000,
    ) -> None:
        self.name = name
        self.grid_params = grid_params
        self.dem_min_m = dem_min_m
        self.dem_max_m = dem_max_m
        self.band_step_ft = band_step_ft
        self.grid = make_grid(**asdict(self.grid_params))
        self.variables = {variable.key: variable for variable in variables}

    @cached_property
    def is_geographic(self) -> bool:
        """Whether cell area varies across the grid (geographic CRS) or is
        constant (projected/linear CRS). Drives whether an area raster is
        needed."""
        return CRS.from_user_input(self.grid_params.crs).is_geographic

    @cached_property
    def cell_area(self) -> float:
        """The constant per-cell area, in the grid CRS's units. Only meaningful
        on a projected grid; raises on a geographic grid, where area varies by
        latitude and an area raster is required instead."""
        if self.is_geographic:
            raise ValueError(
                f'{self.name}: cell_area is constant only on a projected grid; '
                'this grid is geographic, so per-cell area varies by latitude.',
            )
        return abs(self.grid.base_grid[0, 0].area)
