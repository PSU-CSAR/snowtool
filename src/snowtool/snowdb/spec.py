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

from functools import cached_property
from typing import TYPE_CHECKING

from pyproj import CRS

from snowtool.snowdb.config import ZoneLayerParams
from snowtool.snowdb.grid import GridParams, make_grid

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pydantic import BaseModel
    from shapely import Geometry

    from snowtool.snowdb.config import DatasetConfig
    from snowtool.snowdb.coverage import CoverageDomain
    from snowtool.snowdb.ingest import Ingester
    from snowtool.snowdb.variables import DatasetVariable

# ``GridParams`` lives in ``grid.py`` (it is the grid's parameter set) but is
# re-exported here, the canonical import path for a dataset's definition types.
__all__ = ['DEFAULT_ZONES', 'DatasetSpec', 'GridParams', 'ZoneConfig']


# A dataset's zone configuration: provider name -> layer key -> the default
# query params for that layer (a :class:`ZoneLayerParams`). A provider's presence
# here *enables* it for the dataset (it is generated and served); absence means
# the dataset has no such zone layer.
ZoneConfig = dict[str, dict[str, ZoneLayerParams]]

# The zones every standard dataset enables, with their behaviour-preserving
# defaults (1000 ft elevation bands; a 50% forest threshold). A dataset that
# wants a subset (or different params) overrides this.
DEFAULT_ZONES: ZoneConfig = {
    'terrain': {'elevation': ZoneLayerParams(band_step_ft=1000)},
    'landcover': {'forest_cover': ZoneLayerParams(threshold_pct=50)},
}


class DatasetSpec:
    def __init__(
        self,
        name: str,
        *,
        grid_params: GridParams,
        variables: Iterable[DatasetVariable] = (),
        zones: ZoneConfig | None = None,
        ingester: Ingester | None = None,
        footprint: Geometry | None = None,
    ) -> None:
        self.name = name
        self.grid_params = grid_params
        # The zone layers this dataset enables and their per-layer default query
        # params (band step, forest threshold, ...). A provider listed here is
        # generated + served for this dataset; one absent is not. Defaults to the
        # standard terrain + land-cover set (see DEFAULT_ZONES).
        self.zones: ZoneConfig = DEFAULT_ZONES if zones is None else zones
        self.grid = make_grid(**self.grid_params.model_dump())
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

    def enables(self: DatasetSpec, provider_name: str) -> bool:
        """Whether this dataset enables (generates + serves) ``provider_name``."""
        return provider_name in self.zones

    def zone_params(
        self: DatasetSpec,
        provider_name: str,
        layer_key: str,
    ) -> ZoneLayerParams:
        """The configured default query params for one zone layer.

        An all-``None`` :class:`ZoneLayerParams` when the provider/layer is not
        configured, so callers read params uniformly without a presence check.
        """
        return self.zones.get(provider_name, {}).get(layer_key, ZoneLayerParams())

    @classmethod
    def from_config(
        cls: type[DatasetSpec],
        config: DatasetConfig,
        name: str,
    ) -> DatasetSpec:
        """Deserialize a :class:`~snowtool.snowdb.config.DatasetConfig` into a spec.

        A trivial pass-through (no merge, no runtime kind): the config's grid,
        variables, ``zones`` and ``footprint`` are already the domain types, so
        they carry straight over; only the ``ingester`` *name* is resolved to the
        concrete ingester from the registry (``None`` for a read-only/derived
        dataset). ``name`` is supplied separately because the config does not carry
        one -- it comes from where the config is registered.
        """
        from snowtool.snowdb.datasets import INGESTERS

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
        return cls(
            name,
            grid_params=config.grid,
            variables=list(config.variables.values()),
            zones=config.zones,
            ingester=ingester,
            footprint=config.footprint,
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
        constant (projected/linear CRS). Drives whether an AOI raster burns
        per-row geodesic area or the constant ``cell_area``."""
        return self.crs.is_geographic

    @cached_property
    def cell_area(self) -> float:
        """The constant per-cell area, in square metres. Only meaningful on a
        projected grid; raises on a geographic grid, where area varies by
        latitude and the AOI raster burns per-row geodesic area instead.

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
        from snowtool.snowdb.zonal_stat_models import build_zonal_stat_model

        return build_zonal_stat_model(self)

    @cached_property
    def zonal_stats_model(self) -> type[BaseModel]:
        """The generated per-date response model (a ``date`` plus its zones)."""
        from snowtool.snowdb.zonal_stat_models import build_zonal_stats_model

        return build_zonal_stats_model(self, self.zonal_stat_model)
