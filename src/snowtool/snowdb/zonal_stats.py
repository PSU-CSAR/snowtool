from __future__ import annotations

import asyncio
import csv
import math

from dataclasses import dataclass
from datetime import date
from typing import IO, TYPE_CHECKING, Self

import numpy
import numpy.typing

from snowtool.exceptions import QueryParameterError
from snowtool.snowdb.aoi_raster import AOIRaster
from snowtool.snowdb.raster import DataRaster
from snowtool.snowdb.raster.collection import RasterCollection
from snowtool.snowdb.variables import DatasetVariable, Reducer
from snowtool.snowdb.zones.zone_layer import available_zones
from snowtool.snowdb.zones.zoning import Zone

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pydantic import BaseModel

    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.raster.tiff_cache import TiffCache
    from snowtool.snowdb.spec import DatasetSpec
    from snowtool.snowdb.zones.zone_layer import AvailableZone
    from snowtool.snowdb.zones.zoning import ZoneScheme

# Cap on the crossed product size (number of cells = rows in the CSV / objects in
# the JSON, and the cell axis of the in-memory array). Crossing several
# fine-grained axes multiplies their zone counts, so a query is rejected before any
# raster is read if its product would exceed this. The HTTP/CLI layer can pass a
# settings-derived override; this is the library default.
DEFAULT_MAX_ZONE_CELLS = 10_000


@dataclass(frozen=True)
class ZoneSelection:
    """One axis of a crossed-zone query: a zone layer + an optional scheme override.

    ``layer_key`` is a registry key (``'<provider>.<layer.key>'``, e.g.
    ``'terrain.elevation'``). ``override`` is the axis' single scheme param -- a
    band step (banded layers) or a split threshold (threshold layers); ``None``
    uses the scheme default. The scheme owns what the value means and how it is
    parsed (see :meth:`ZoneScheme.parse_override`/:meth:`ZoneScheme.with_override`),
    so a categorical axis simply takes no override.
    """

    layer_key: str
    override: int | float | None = None


def _unknown_layer(
    layer_key: str,
    registry: Mapping[str, AvailableZone],
) -> QueryParameterError:
    """A uniform 'unknown zone layer' error listing the available registry keys."""
    return QueryParameterError(
        f'Unknown zone layer {layer_key!r}; available: '
        f'{", ".join(sorted(registry)) or "(none)"}.',
    )


def parse_zone_selection(
    token: str,
    registry: Mapping[str, AvailableZone],
) -> ZoneSelection:
    """Parse a ``LAYER[:override]`` token into a :class:`ZoneSelection`.

    ``LAYER`` is a registry key (``'<provider>.<layer.key>'``); an optional
    ``:override`` sets the axis' scheme param. The token is delegated to the
    layer's scheme (:meth:`ZoneScheme.parse_override`), which types it (a band
    step, a split threshold) or rejects it (a categorical axis takes none).
    Backs the CLI ``--zone`` flag; raises a clean error (listing the choices) on
    an unknown layer.
    """
    layer_key, sep, raw = token.partition(':')
    available = registry.get(layer_key)
    if available is None:
        raise _unknown_layer(layer_key, registry)
    if not sep:
        return ZoneSelection(layer_key)
    return ZoneSelection(layer_key, available.scheme.parse_override(layer_key, raw))


@dataclass
class Result:
    """One (date, crossed-zone cell, variable) reduction.

    ``zone`` is the cell's per-axis zone tuple (one :class:`Zone` per selected
    layer, in selection order).
    """

    date: date
    zone: tuple[Zone, ...]
    variable: DatasetVariable
    value: float
    area: float


class ZonalStats:
    def __init__(
        self: Self,
        spec: DatasetSpec,
        variables: set[DatasetVariable],
        zone_layers: tuple[str, ...],
        zone_cells: tuple[tuple[Zone, ...], ...],
        dates: tuple[date, ...],
        *results: Result,
    ) -> None:
        self.spec = spec
        # The crossed zone axes (registry keys, in selection order) and the flat
        # list of product cells (each a per-axis zone tuple, in mixed-radix order).
        self.zone_layers = zone_layers
        self._variables_index = {
            variable: idx + 1
            for idx, variable in enumerate(
                sorted(variables, key=lambda v: v.key),
            )
        }
        # Cells arrive in flat product order from the zone index; the index
        # preserves that order rather than re-sorting.
        self._cells_index = {cell: idx for idx, cell in enumerate(zone_cells)}
        self._dates_index = {dt: idx for idx, dt in enumerate(sorted(dates))}
        # float64, not float32: the per-cell reduction runs in float64
        # (_ZoneIndex.reduce), and area/total stats reach ~1e9-scale values that
        # float32 truncates to ~7 significant digits. The array is tiny
        # (cells x dates x stats) and JSON output is float64 anyway, so store the
        # full precision rather than round-tripping through float32.
        self._array = numpy.full(
            (
                len(self._dates_index),
                len(self._cells_index),
                len(self._variables_index) + 1,
            ),
            -numpy.inf,
            dtype=numpy.float64,
        )

        self.add_results(*results)

    @property
    def n_cells(self: Self) -> int:
        """The crossed-zone product size (cells per date); 1 for a whole-basin
        (K=0) query."""
        return len(self._cells_index)

    def add_result(self: Self, result: Result) -> None:
        cell = self._array[self._dates_index[result.date]][
            self._cells_index[result.zone]
        ]

        cell[0] = result.area
        cell[self._variables_index[result.variable]] = result.value

    def add_results(self: Self, *results: Result) -> None:
        for result in results:
            self.add_result(result)

    def validate(self: Self) -> None:
        if (self._array == -numpy.inf).any():
            raise ValueError(
                'Results array is incomplete. '
                'Ensure all data was processed and added to results successfully.',
            )

    def _zone_stats(self: Self, date_idx: int, cell_idx: int) -> dict[str, float]:
        """The scaled per-cell stat values (``area_m2`` + each variable) for one
        (date, cell) -- the single source the JSON (:meth:`dump`) and CSV
        (:meth:`dump_to_csv`) serializers share, so both apply the same unit
        scaling and ``float`` coercion. The keys are ordered ``area_m2`` first
        then the variables in ``_variables_index`` order. A cell with no valid
        pixels carries ``nan``, which each serializer renders as its own 'missing'
        token (JSON null / empty cell).
        """
        cell = self._array[date_idx][cell_idx]
        values = {'area_m2': float(cell[0])}
        for variable, var_idx in self._variables_index.items():
            values[variable.stat_name] = float(variable.unit.scale(cell[var_idx]))
        return values

    @staticmethod
    def _zone_refs(
        layers: tuple[str, ...],
        cell: tuple[Zone, ...],
    ) -> list[BaseModel]:
        """Self-describing per-axis zone refs for one crossed-zone cell.

        Each :class:`Zone` builds its own concrete ``ZoneRef`` (:meth:`Zone.ref`),
        so a new zone kind owns its ref construction with no change here.
        """
        return [zone.ref(layer) for layer, zone in zip(layers, cell, strict=True)]

    def dump(self: Self) -> list[BaseModel]:
        self.validate()
        stat_model = self.spec.zonal_stat_model
        stats_model = self.spec.zonal_stats_model
        # Zone refs depend only on the cell, not the date; build them once (one
        # pydantic validation per cell) and reuse across every date.
        cell_refs = [
            self._zone_refs(self.zone_layers, cell) for cell in self._cells_index
        ]
        stats: list[BaseModel] = []
        for date_, date_idx in self._dates_index.items():
            cells: list[BaseModel] = []
            for cell_idx in range(len(self._cells_index)):
                cells.append(
                    stat_model(
                        zone=cell_refs[cell_idx],
                        **self._zone_stats(date_idx, cell_idx),
                    ),
                )
            stats.append(
                stats_model(
                    date=date_,
                    zone_layers=list(self.zone_layers),
                    zones=cells,
                ),
            )
        return stats

    def _axis_kinds(self: Self) -> tuple[Zone, ...]:
        """A sample zone per axis, to type the CSV columns (header + row layout).

        Every cell shares the same per-axis zone kind (one scheme per axis), so any
        cell is a faithful template. There is always at least one cell (every scheme
        yields at least one zone).
        """
        return next(iter(self._cells_index))

    def dump_to_csv(self: Self, out: IO) -> None:
        self.validate()
        writer = csv.writer(out, quoting=csv.QUOTE_MINIMAL)

        # One row per (date, crossed-zone cell). Each axis describes its own
        # columns (:meth:`Zone.csv_columns`): a structured axis (banded/threshold)
        # expands to two typed, unit-bearing columns, a categorical axis to one.
        # The header comes from a sample cell's columns, every row from its own.
        # Then area + each variable.
        sample = self._axis_kinds()
        headers: list[str] = ['date']
        for layer, zone in zip(self.zone_layers, sample, strict=True):
            headers.extend(header for header, _ in zone.csv_columns(layer))
        headers.append('area_m2')
        headers.extend(variable.stat_name for variable in self._variables_index)
        writer.writerow(headers)

        # The zone columns depend only on the cell, not the date; format them once
        # per cell and reuse across every date's row.
        cell_columns = [
            [
                value
                for layer, zone in zip(self.zone_layers, cell, strict=True)
                for _, value in zone.csv_columns(layer)
            ]
            for cell in self._cells_index
        ]

        for date_, date_idx in self._dates_index.items():
            for cell_idx in range(len(self._cells_index)):
                row: list[str] = [date_.isoformat(), *cell_columns[cell_idx]]
                # Empty cell for a no-data reduction (nan), matching dump()'s JSON
                # null -- never the literal 'nan'.
                row.extend(
                    '' if math.isnan(value) else str(value)
                    for value in self._zone_stats(date_idx, cell_idx).values()
                )
                writer.writerow(row)

    @classmethod
    async def calculate(
        cls: type[Self],
        aoi: AOIRaster,
        rasters: RasterCollection,
        cache: TiffCache,
        dataset: Dataset,
        zone_selections: Sequence[ZoneSelection] = (),
        *,
        max_zone_cells: int = DEFAULT_MAX_ZONE_CELLS,
    ) -> Self:
        """Reduce ``rasters`` over the AOI, crossed by the selected zone layers.

        ``zone_selections`` names the zone-layer axes to cross (each resolved
        against ``dataset``'s zone layers + the provider registry). An **empty**
        selection means *no* stratification: the reduction is over the whole basin,
        producing a single cell per date whose ``zone`` tuple is empty (the K=0
        case of the crossed index). Each selected zone layer is read live, windowed
        to the AOI, and assigned to per-pixel ordinals; the crossed index is the
        cartesian product of the axes. A query whose product would exceed
        ``max_zone_cells`` is rejected before any raster is read.
        """
        spec = dataset.spec
        selections = list(zone_selections)

        # Resolve each axis (registry + per-selection scheme overrides). The zone
        # geometry (which pixel is in which crossed cell, and each cell's total
        # area) depends only on the AOI mask + the zone layers -- not on any
        # variable or date -- so it is computed once here and reused by every
        # reduction.
        registry = available_zones(dataset.providers.values())
        resolved: list[tuple[AvailableZone, ZoneScheme]] = []
        for selection in selections:
            available = registry.get(selection.layer_key)
            if available is None:
                raise _unknown_layer(selection.layer_key, registry)
            # Fold the dataset's configured params for this layer into a configured
            # scheme, then apply the selection's explicit override (if any). After
            # this the scheme carries everything; zones()/assign() take no kwargs.
            scheme = available.scheme.configured(
                spec.zone_params(available.provider.name, available.layer.key),
            )
            if selection.override is not None:
                scheme = scheme.with_override(selection.override)
            resolved.append((available, scheme))

        # The axes' zones (hence the crossed product size) are known from the
        # schemes alone, with no raster reads -- so guard against a runaway product
        # before paying for any I/O.
        axes: list[tuple[Zone, ...]] = [scheme.zones() for _, scheme in resolved]
        n_cells = math.prod(len(axis) for axis in axes)
        if n_cells > max_zone_cells:
            raise QueryParameterError(
                f'crossed zone query would produce {n_cells} cells '
                f'(> max_zone_cells={max_zone_cells}); use fewer axes, coarser '
                'steps, or raise the limit.',
            )

        # Read each selected zone layer live (windowed to the AOI), concurrently.
        async def _read_axis(available: AvailableZone) -> numpy.typing.NDArray:
            layer = available.layer
            values = numpy.full(aoi.array.shape, layer.nodata, dtype=layer.dtype)
            await aoi.load_raster_tiles_into_array(
                dataset.zones[available.provider.name].raster(layer),
                values,
                cache,
            )
            return values

        axis_arrays = await asyncio.gather(
            *(_read_axis(available) for available, _ in resolved),
        )
        ordinals_list = [
            scheme.assign(values)
            for (_, scheme), values in zip(resolved, axis_arrays, strict=True)
        ]
        zone_layers = [selection.layer_key for selection in selections]

        zone_index = _ZoneIndex.build(axes, ordinals_list, aoi.array)

        # Fan out across the raster set; each raster's tile reads fan out
        # further inside _calc. The handle cache dedupes/bounds open COGs.
        per_raster = await asyncio.gather(
            *(
                cls._calc(aoi, variable, raster, zone_index, cache)
                for variable, variable_rasters in rasters.items()
                for raster in variable_rasters
            ),
        )
        results: list[Result] = [
            result for raster_results in per_raster for result in raster_results
        ]

        return cls(
            spec,
            rasters.variables,
            tuple(zone_layers),
            zone_index.cell_zones,
            tuple(rasters.dates),
            *results,
        )

    @staticmethod
    async def _calc(
        aoi: AOIRaster,
        variable: DatasetVariable,
        raster: DataRaster,
        zone_index: _ZoneIndex,
        cache: TiffCache,
    ) -> list[Result]:
        date_ = raster.date
        values_array = numpy.empty_like(aoi.array, dtype=variable.dtype)
        values_array[:] = variable.nodata

        await aoi.load_raster_tiles_into_array(raster, values_array, cache)

        # The reduction runs only over in-zone pixels that actually have data;
        # everything else (zone geometry, cell areas) was precomputed once.
        selection = zone_index.in_zone & (values_array != variable.nodata)
        values = zone_index.reduce(
            variable.reducer,
            values_array,
            aoi.array,
            selection,
        )

        return [
            Result(
                date=date_,
                variable=variable,
                zone=cell,
                value=float(values[idx]),
                area=float(zone_index.areas[idx]),
            )
            for idx, cell in enumerate(zone_index.cell_zones)
        ]


@dataclass
class _ZoneIndex:
    """Per-pixel crossed-zone membership for one AOI, computed once and reused.

    Combines K per-axis ordinal arrays into one **mixed-radix linear index** over
    the product space (size ``prod(dims)``). ``index`` is that combined cell index
    per pixel (meaningful only where ``in_zone``); ``in_zone`` is the boolean of
    pixels that are in every axis' zone *and* in the AOI mask; ``areas[c]`` is
    crossed cell ``c``'s total geographic area. ``cell_zones`` carries the per-axis
    :class:`Zone` tuple for every product cell, in the same flat order.
    """

    axes: list[tuple[Zone, ...]]
    dims: list[int]
    index: numpy.typing.NDArray[numpy.int64]
    in_zone: numpy.typing.NDArray[numpy.bool_]
    areas: numpy.typing.NDArray[numpy.float64]
    cell_zones: tuple[tuple[Zone, ...], ...]

    @classmethod
    def build(
        cls: type[Self],
        axes: list[tuple[Zone, ...]],
        ordinals: list[numpy.typing.NDArray[numpy.int64]],
        area: numpy.typing.NDArray[numpy.float32],
    ) -> Self:
        """Cross K per-axis ordinal arrays into one crossed-cell index.

        ``area`` is the AOI raster: per-pixel cell area inside the basin, 0
        outside -- so it is both the in/out membership signal and the area
        weights. A pixel is in-zone only when every axis assigns it a real ordinal
        (``>= 0``) and it is inside the AOI (``area > 0``); its crossed cell is the
        mixed-radix combination of the per-axis ordinals.
        """
        dims = [len(axis) for axis in axes]
        n = math.prod(dims)
        in_zone = area > 0
        combined = numpy.zeros(area.shape, dtype=numpy.int64)
        for ords, dim in zip(ordinals, dims, strict=True):
            in_zone = in_zone & (ords >= 0)
            # Out-of-zone ordinals (-1) make combined garbage, but those pixels are
            # excluded by in_zone before it is ever read, so the radix math is only
            # consumed where every axis is valid.
            combined = combined * dim + ords
        areas = numpy.bincount(
            combined[in_zone],
            weights=area[in_zone],
            minlength=n,
        ).astype(numpy.float64)
        return cls(
            axes=axes,
            dims=dims,
            index=combined,
            in_zone=in_zone,
            areas=areas,
            cell_zones=cls._enumerate_cells(axes, dims),
        )

    @staticmethod
    def _enumerate_cells(
        axes: list[tuple[Zone, ...]],
        dims: list[int],
    ) -> tuple[tuple[Zone, ...], ...]:
        """The product cells in flat (mixed-radix) order: one Zone tuple per cell."""
        cells: list[tuple[Zone, ...]] = []
        for flat in range(math.prod(dims)):
            zones: list[Zone] = []
            for i, dim in enumerate(dims):
                stride = math.prod(dims[i + 1 :])
                zones.append(axes[i][(flat // stride) % dim])
            cells.append(tuple(zones))
        return tuple(cells)

    def reduce(
        self: Self,
        reducer: Reducer,
        values_array: numpy.typing.NDArray,
        area_array: numpy.typing.NDArray[numpy.float32],
        selection: numpy.typing.NDArray[numpy.bool_],
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Area-weighted reduction for every crossed cell at once, over ``selection``.

        One pass via ``bincount`` over the combined cell index instead of a
        per-cell masked reduction. Area weighting is automatic from the grid CRS
        (``area`` is geodesic on a geographic grid, constant on a projected one), so
        MEAN degenerates to a plain mean when cells are equal-area. A cell with no
        selected pixels is ``nan`` (as a per-pixel empty reduction would be).
        """
        n = math.prod(self.dims)
        idx = self.index[selection]
        values = values_array[selection]
        areas = area_array[selection]
        weighted = numpy.bincount(idx, weights=values * areas, minlength=n).astype(
            numpy.float64,
        )

        match reducer:
            case Reducer.MEAN:
                area_sum = numpy.bincount(idx, weights=areas, minlength=n)
                with numpy.errstate(invalid='ignore', divide='ignore'):
                    result = weighted / area_sum
            case Reducer.TOTAL:
                result = weighted

        # Empty cells divide to nan for MEAN already, but TOTAL needs it set
        # explicitly so a no-data cell reads nan rather than a spurious 0.
        result[numpy.bincount(idx, minlength=n) == 0] = numpy.nan
        return result
