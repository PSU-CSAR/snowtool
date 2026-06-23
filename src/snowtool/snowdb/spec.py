"""Dataset specifications: the path-independent *definition* of a dataset kind.

A :class:`DatasetSpec` is the "SNODAS-ness" of a dataset — its grid and its
variables (and ingest). Elevation bands span a single global range shared by
every dataset (:data:`~snowtool.snowdb.constants.MIN_ELEVATION_M` /
``MAX_ELEVATION_M``), not a per-dataset DEM range, so they stay comparable across
AOIs and datasets. The built-in specs
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

    from pydantic import BaseModel
    from shapely import Geometry

    from snowtool.snowdb.config import DatasetConfig
    from snowtool.snowdb.coverage import CoverageDomain
    from snowtool.snowdb.ingest import Ingester
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
        variables: Iterable[DatasetVariable] = (),
        band_step_ft: int = 1000,
        ingester: Ingester | None = None,
        footprint: Geometry | None = None,
    ) -> None:
        self.name = name
        self.grid_params = grid_params
        self.band_step_ft = band_step_ft
        self.grid = make_grid(**asdict(self.grid_params))
        self.variables = {variable.key: variable for variable in variables}
        # How this dataset kind turns a source artifact into per-date COGs;
        # None means the dataset has no ingest (e.g. read-only / derived). See
        # snowtool.snowdb.ingest.
        self.ingester = ingester
        # The region this dataset actually serves, as a single (multi)polygon in
        # the grid CRS -- e.g. a MODIS block minus a never-ingested tile. Static
        # (grid definition), not per-date. None means the dataset serves its whole
        # grid extent, so coverage defaults to the extent rectangle. See
        # CoverageDomain.
        self.footprint = footprint

    @classmethod
    def from_config(
        cls: type[DatasetSpec],
        config: DatasetConfig,
        name: str,
    ) -> DatasetSpec:
        """Deserialize a :class:`~snowtool.snowdb.config.DatasetConfig` into a spec.

        A trivial field map (no merge, no runtime kind): the config's grid,
        variables, ``band_step_ft`` and ``footprint`` are reconstructed as-is, and
        its ``ingester`` *name* is resolved to the concrete ingester from the
        registry (``None`` for a read-only/derived dataset). ``name`` is supplied
        separately because the config does not carry one -- it comes from where the
        config is registered.
        """
        import shapely

        from snowtool.snowdb.datasets import INGESTERS
        from snowtool.snowdb.variables import DatasetVariable, Unit

        grid_params = GridParams(**config.grid.model_dump())
        variables = [
            DatasetVariable(
                key=key,
                unit=Unit(name=var.unit.name, scale_factor=var.unit.scale_factor),
                reducer=var.reducer,
                dtype=var.dtype,
                nodata=var.nodata,
                glob=var.glob,
            )
            for key, var in config.variables.items()
        ]
        if config.ingester is None:
            ingester = None
        elif config.ingester in INGESTERS:
            ingester = INGESTERS[config.ingester]
        else:
            known = ', '.join(sorted(INGESTERS)) or '(none)'
            raise ValueError(
                f'{name!r}: unknown ingester {config.ingester!r}. '
                f'Known ingesters: {known}.',
            )
        footprint = (
            shapely.geometry.shape(config.footprint)
            if config.footprint is not None
            else None
        )
        return cls(
            name,
            grid_params=grid_params,
            variables=variables,
            band_step_ft=config.band_step_ft,
            ingester=ingester,
            footprint=footprint,
        )

    @cached_property
    def crs(self) -> CRS:
        """The grid's CRS (pyproj), the single source for every CRS-derived
        value -- ``is_geographic``, ``cell_area``, and the dataset's rasterio
        write CRS (:attr:`Dataset.grid_crs`)."""
        # make_grid always sets a CRS (GridParams.crs defaults to WGS84), so this
        # narrows the Optional griffine exposes on grid.crs.
        crs = self.grid.crs
        if crs is None:  # pragma: no cover - defensive; make_grid always sets one
            raise ValueError(f'{self.name}: grid has no CRS')
        return crs

    @cached_property
    def coverage_domain(self) -> CoverageDomain:
        """The static region this dataset can serve.

        Used by AOI coverage classification: the dataset's ``footprint`` when it
        declares one (e.g. a MODIS block minus a never-ingested tile), else the
        full grid-extent rectangle -- so a basin over a permanently-empty hole is
        not reported as fully covered.
        """
        from snowtool.snowdb.coverage import CoverageDomain

        return CoverageDomain.from_grid(self.grid, footprint=self.footprint)

    @cached_property
    def is_geographic(self) -> bool:
        """Whether cell area varies across the grid (geographic CRS) or is
        constant (projected/linear CRS). Drives whether an area raster is
        needed."""
        return self.crs.is_geographic

    @cached_property
    def cell_area(self) -> float:
        """The constant per-cell area, in square metres. Only meaningful on a
        projected grid; raises on a geographic grid, where area varies by
        latitude and an area raster is required instead.

        griffine reports a projected grid's planar cell area in the CRS's own
        linear units squared, so it is converted to m^2 here -- every area we
        emit (the ``area_m2`` field, the CSV column) is metres regardless of the
        grid's units."""
        if self.is_geographic:
            raise ValueError(
                f'{self.name}: cell_area is constant only on a projected grid; '
                'this grid is geographic, so per-cell area varies by latitude.',
            )
        planar_area = abs(self.grid.base_grid[0, 0].area)
        meters_per_unit = self.crs.axis_info[0].unit_conversion_factor
        return planar_area * meters_per_unit**2

    @cached_property
    def model_prefix(self) -> str:
        """CamelCase prefix for this dataset's generated response models
        (e.g. ``snodas`` -> ``Snodas`` -> ``SnodasZonalStat``).

        Names that differ only by case or ``-``/``_`` collapse to the same
        prefix (``foo-bar`` and ``foo_bar`` both -> ``FooBar``), so SnowDb
        enforces prefix uniqueness across its specs to avoid OpenAPI
        schema-name collisions between datasets."""
        return ''.join(
            part.capitalize() for part in self.name.replace('-', '_').split('_')
        )

    @cached_property
    def zonal_stat_model(self) -> type[BaseModel]:
        """The generated per-elevation-band response model for this dataset."""
        from snowtool.snowdb.response_models import build_zonal_stat_model

        return build_zonal_stat_model(self)

    @cached_property
    def zonal_stats_model(self) -> type[BaseModel]:
        """The generated per-date response model (a ``date`` plus its zones)."""
        from snowtool.snowdb.response_models import build_zonal_stats_model

        return build_zonal_stats_model(self, self.zonal_stat_model)
