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

import copy

from functools import cached_property
from typing import TYPE_CHECKING

from pyproj import CRS

from snowtool.snowdb.config import (
    BandStepParams,
    BucketParams,
    EntropyThresholdParams,
    ThresholdParams,
    ZoneLayerParams,
)
from snowtool.snowdb.grid import GridParams, make_grid

if TYPE_CHECKING:
    from collections.abc import Iterable

    from geojson_pydantic.geometries import Geometry

    from snowtool.snowdb.config import DatasetConfig
    from snowtool.snowdb.coverage import CoverageDomain
    from snowtool.snowdb.ingest import Ingester
    from snowtool.snowdb.variables import DatasetVariable

# ``GridParams`` lives in ``grid.py`` (it is the grid's parameter set) but is
# re-exported here, the canonical import path for a dataset's definition types.
__all__ = ['DEFAULT_ZONES', 'DatasetSpec', 'GridParams', 'ZoneConfig']


# A dataset's zone configuration: provider name -> layer key -> the default
# query params for that layer (a :class:`ZoneLayerParams`, or ``None`` for a
# layer enabled with no params). A provider's presence here *enables* it for the
# dataset (it is generated and served); absence means the dataset has no such
# zone layer.
ZoneConfig = dict[str, dict[str, ZoneLayerParams | None]]

# The zones every standard dataset enables: it enumerates every served layer
# across the built-in providers (terrain's five, land cover's one), each with its
# behaviour-preserving defaults (these values equal the scheme's own defaults --
# 1000 ft elevation bands, 4 northness/eastness buckets, a 0.5 aspect-entropy
# threshold, a 50% forest threshold; aspect is categorical and maps to ``None``,
# taking no param). A dataset that wants a subset (or different params)
# overrides this.
DEFAULT_ZONES: ZoneConfig = {
    'terrain': {
        'elevation': BandStepParams(band_step_ft=1000),
        'aspect': None,
        'northness': BucketParams(buckets=4),
        'eastness': BucketParams(buckets=4),
        'aspect_entropy': EntropyThresholdParams(entropy_threshold=0.5),
    },
    'landcover': {'forest_cover': ThresholdParams(threshold_pct=50)},
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
        # standard terrain + land-cover set (see DEFAULT_ZONES), deep-copied so no
        # spec aliases (and could mutate) the shared module-level dict -- the
        # values are small frozen models, so the copy is cheap and runs once per
        # spec construction.
        self.zones: ZoneConfig = (
            copy.deepcopy(DEFAULT_ZONES) if zones is None else zones
        )
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
    ) -> ZoneLayerParams | None:
        """The configured default query params for one zone layer.

        ``None`` when the provider/layer is not configured, or is configured
        with no params -- either way the scheme's own defaults apply, so
        callers pass the result to :meth:`ZoneScheme.configured` uniformly.
        """
        return self.zones.get(provider_name, {}).get(layer_key)

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
        import shapely

        from snowtool.snowdb.coverage import CoverageDomain

        # The footprint is a geojson-pydantic geometry (grid CRS); the geometry
        # math wants shapely, so convert once here (geojson-pydantic exposes
        # __geo_interface__, so shapely reads it directly).
        footprint = (
            shapely.geometry.shape(self.footprint)
            if self.footprint is not None
            else None
        )
        return CoverageDomain.from_grid(self.grid, footprint=footprint)

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
