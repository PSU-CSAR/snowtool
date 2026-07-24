from __future__ import annotations

import asyncio
import csv
import io
import itertools
import math

from dataclasses import dataclass
from datetime import date
from typing import IO, TYPE_CHECKING, Self, assert_never

import numpy
import numpy.typing

from snowtool.exceptions import QueryParameterError, ZoneParamsError
from snowtool.snowdb.aoi_raster import AOIRaster
from snowtool.snowdb.raster import DataRaster
from snowtool.snowdb.raster.collection import RasterCollection
from snowtool.snowdb.variables import DatasetVariable, Reducer
from snowtool.snowdb.zonal_stat_models import CompactStats, CompactZone
from snowtool.snowdb.zones.zone_layer import available_zones
from snowtool.snowdb.zones.zoning import CategoricalZoneDescription, Zone

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.raster.tiff_cache import TiffCache
    from snowtool.snowdb.spec import DatasetSpec
    from snowtool.snowdb.zonal_stat_models import ZoneRef
    from snowtool.snowdb.zones.zone_layer import AvailableZone
    from snowtool.snowdb.zones.zoning import ZoneScheme

# Cap on the crossed product size (number of cells = rows in the CSV / objects in
# the JSON, and the cell axis of the in-memory array). Crossing several
# fine-grained axes multiplies their zone counts, so a query is rejected before any
# raster is read if its product would exceed this. The HTTP/CLI layer can pass a
# settings-derived override; this is the library default.
DEFAULT_MAX_ZONE_CELLS = 10_000

# Cap on how many per-raster reductions (`_calc`) run concurrently. Each in-flight
# reduction holds a transient full-window value array and issues its own unbounded
# tile-fetch batch, so a wide date range would otherwise allocate one such window
# per date at once. The semaphore bounds peak memory and fetch fan-out without
# changing results (each reduction writes a disjoint array slice). The HTTP/CLI
# layer can pass a settings-derived override; this is the library default.
DEFAULT_MAX_CONCURRENT_RASTERS = 16


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


def resolve_zone_axis(
    selection: ZoneSelection,
    registry: Mapping[str, AvailableZone],
    spec: DatasetSpec,
) -> tuple[AvailableZone, ZoneScheme]:
    """Resolve one query axis to its registry entry and fully configured scheme.

    Looks ``selection.layer_key`` up in ``registry`` (raising a uniform
    :class:`QueryParameterError` on an unknown layer), folds the dataset's
    configured params for that layer into a configured scheme, then applies the
    selection's explicit override (if any). After this the scheme carries
    everything; ``zones()``/``assign()`` take no kwargs. A dataset-config
    :class:`ZoneParamsError` is re-wrapped to name the offending ``zones`` entry.
    """
    available = registry.get(selection.layer_key)
    if available is None:
        raise _unknown_layer(selection.layer_key, registry)
    try:
        scheme = available.scheme.configured(
            spec.zone_params(available.provider.name, available.layer.key),
        )
    except ZoneParamsError as e:
        raise ZoneParamsError(
            f'dataset {spec.name!r} zones'
            f'[{available.provider.name!r}][{available.layer.key!r}]: {e}',
        ) from e
    if selection.override is not None:
        scheme = scheme.with_override(selection.override)
    return available, scheme


def parse_zone_selection(
    token: str,
    registry: Mapping[str, AvailableZone],
) -> ZoneSelection:
    """Parse a ``LAYER[:PARAM=VALUE]`` token into a :class:`ZoneSelection`.

    ``LAYER`` is a registry key (``'<provider>.<layer.key>'``). With no ``:`` the
    layer takes its scheme default. ``:PARAM=VALUE`` overrides the layer's single
    scheme parameter, where ``PARAM`` is exactly the override key the dataset
    resource advertises for that layer (e.g. ``terrain.elevation:band_step_ft=500``);
    the value is delegated to the layer's scheme (:meth:`ZoneScheme.parse_override`),
    which types it. Backs the CLI ``--zone`` flag and the API's ``zone`` query
    tokens; raises a clean :class:`QueryParameterError` on an unknown layer, a
    malformed override, an unknown ``PARAM``, or an override on a categorical layer.
    """
    layer_key, sep, rest = token.partition(':')
    available = registry.get(layer_key)
    if available is None:
        raise _unknown_layer(layer_key, registry)
    if not sep:
        return ZoneSelection(layer_key)

    param, eq, raw = rest.partition('=')
    if not eq:
        raise QueryParameterError(
            f'Malformed zone override {token!r}; expected LAYER:PARAM=VALUE, '
            f'e.g. terrain.elevation:band_step_ft=500.',
        )
    desc = available.scheme.describe()
    if isinstance(desc, CategoricalZoneDescription):
        raise QueryParameterError(
            f'Zone layer {layer_key!r} is categorical and takes no override.',
        )
    if param != desc.param:
        raise QueryParameterError(
            f'Unknown override {param!r} for {layer_key!r}; its override is '
            f'{desc.param!r} (e.g. {layer_key}:{desc.param}=<value>).',
        )
    return ZoneSelection(layer_key, available.scheme.parse_override(layer_key, raw))


class ZonalStats:
    def __init__(
        self: Self,
        variables: set[DatasetVariable],
        zone_layers: tuple[str, ...],
        zone_cells: tuple[tuple[Zone, ...], ...],
        areas: numpy.typing.NDArray[numpy.float64],
        dates: tuple[date, ...],
    ) -> None:
        # The crossed zone axes (registry keys, in selection order) and the flat
        # list of product cells (each a per-axis zone tuple, in mixed-radix order).
        self.zone_layers = zone_layers
        self._variables_index = {
            variable: idx
            for idx, variable in enumerate(
                sorted(variables, key=lambda v: v.key),
            )
        }
        # Per-variable reporting scale, aligned with the variable (last) axis of
        # ``_array``. ``Unit.scale`` divides by ``scale_factor``; holding the
        # divisors as one vector lets the serializers scale a whole (date, cell)
        # value block by broadcast instead of scalar-by-scalar in a Python loop.
        self._scale_factors = numpy.array(
            [variable.unit.scale_factor for variable in self._variables_index],
            dtype=numpy.float64,
        )
        # Cells arrive in flat product order from the zone index; keep that order
        # rather than re-sorting.
        self._cells = zone_cells
        # Cell areas are a property of the crossed-zone geometry (the AOI mask + the
        # zone layers) -- date- and variable-invariant -- so they are held once here
        # as a cell-aligned vector, not repeated per (date, variable) in the value
        # array. This is also what lets a zero-date query still report its zones with
        # their areas.
        self._areas = areas
        self._dates_index = {dt: idx for idx, dt in enumerate(sorted(dates))}
        # float64, not float32: area/total stats reach ~1e9-scale values that
        # float32 truncates to ~7 significant digits, and the array is tiny.
        #
        # Filled slice-by-slice via :meth:`fill` -- one whole (date, variable)
        # cell-vector per assignment. Completeness is a construction invariant
        # (``RasterCollection.validate`` guarantees every raster exists), so every
        # slice is written exactly once and 0.0 is never read back unwritten.
        self._array = numpy.zeros(
            (
                len(self._dates_index),
                len(self._cells),
                len(self._variables_index),
            ),
            dtype=numpy.float64,
        )

    @property
    def n_cells(self: Self) -> int:
        """The crossed-zone product size (cells per date); 1 for a whole-basin
        (K=0) query."""
        return len(self._cells)

    def fill(
        self: Self,
        date_: date,
        variable: DatasetVariable,
        values: numpy.typing.NDArray[numpy.float64],
    ) -> None:
        """Write one (date, variable) reduction into the array, vectorized.

        ``values`` is a cell-vector aligned with the flat product (cell) order.
        Cell areas are date-invariant and held once on ``self._areas`` (set at
        construction), so only the variable slice is written here.
        """
        date_idx = self._dates_index[date_]
        self._array[date_idx, :, self._variables_index[variable]] = values

    # The single serialization order both bodies obey: ``area_m2`` first, then each
    # variable's reduced stat in ``_variables_index`` order. Serializers read the
    # scaled value array (:meth:`_scaled_variables`) and ``_areas`` in exactly this
    # order, so JSON and CSV stay byte-identical without a per-cell dict.
    @property
    def _variable_columns(self: Self) -> list[str]:
        """The variable stat-name columns, in ``_variables_index`` order."""
        return [variable.stat_name for variable in self._variables_index]

    def _scaled_variables(self: Self) -> numpy.typing.NDArray[numpy.float64]:
        """The whole value array with each variable's reporting scale applied once.

        ``Unit.scale`` divides by ``scale_factor``; dividing the ``(date, cell,
        variable)`` block by the aligned ``_scale_factors`` vector (broadcast over
        the variable axis) is the identical float64 op done once for the whole
        result rather than scalar-by-scalar per (date, cell) in a Python loop.
        A no-data cell stays ``nan`` (nan / x is nan), which each serializer renders
        as its own 'missing' token (JSON null / empty cell).
        """
        return self._array / self._scale_factors

    @staticmethod
    def _zone_refs(
        layers: tuple[str, ...],
        cell: tuple[Zone, ...],
    ) -> list[ZoneRef]:
        """Self-describing per-axis zone refs for one crossed-zone cell.

        Each :class:`Zone` builds its own concrete ``ZoneRef`` (:meth:`Zone.ref`),
        so a new zone kind owns its ref construction with no change here.
        """
        return [zone.ref(layer) for layer, zone in zip(layers, cell, strict=True)]

    def _emitted_cells(self: Self, *, include_empty_zones: bool) -> list[int]:
        """The cell indices (flat product order) the serializers should emit.

        A crossed-zone cell whose area is 0 has no AOI pixels in that combination
        (the crossed zone doesn't occur in this basin), so it reduces to ``nan`` for
        every variable and date. Crossing several fine axes makes such empty cells
        the combinatoric majority of the product, so by default they are dropped from
        the output; ``include_empty_zones`` keeps the full product. Cell areas are
        date-invariant (held on ``self._areas``), so emptiness is a property of the
        zone geometry alone -- decided the same way whether or not the query matched
        any dates. The whole-basin (K=0) cell always has area and is never dropped.
        """
        return [
            i
            for i in range(len(self._cells))
            if include_empty_zones or self._areas[i] > 0
        ]

    def dump_compact(
        self: Self,
        *,
        include_empty_zones: bool = False,
    ) -> CompactStats:
        """The normalized compact body (zones/variables once, date -> matrix).

        Reads the once-scaled value array (:meth:`_scaled_variables`) and
        ``_areas`` in the shared :attr:`_variable_columns` order that
        :meth:`iter_csv` also uses, so JSON and CSV stay byte-identical. ``area_m2``
        is date-invariant, so it is read straight from ``self._areas`` and hoisted
        into the zone definition (the compact matrix carries only the variables). A
        zero-date query still reports its zones (with their areas) -- ``results`` is
        just an empty matrix; ``zone_layers`` and ``variables`` are always reported.
        """
        cells = self._cells
        variables = self._variable_columns
        emitted = self._emitted_cells(include_empty_zones=include_empty_zones)

        zones: list[CompactZone] = [
            CompactZone(
                zone=self._zone_refs(self.zone_layers, cells[cell_idx]),
                area_m2=float(self._areas[cell_idx]),
            )
            for cell_idx in emitted
        ]

        # Scale every (date, cell, variable) value once up front, then slice.
        scaled = self._scaled_variables()
        results: dict[date, list[list[float | None]]] = {
            date_: [
                [float(v) for v in scaled[date_idx, cell_idx]] for cell_idx in emitted
            ]
            for date_, date_idx in self._dates_index.items()
        }

        return CompactStats(
            zone_layers=list(self.zone_layers),
            variables=variables,
            zones=zones,
            results=results,
        )

    def iter_csv(self: Self, *, include_empty_zones: bool = False) -> Iterator[str]:
        """Return an iterator over CSV chunks for this result (one header or row each).

        Emits the same shared value order :meth:`dump_compact` uses -- ``area_m2``
        then each variable in :attr:`_variable_columns` order, read from ``_areas``
        and the once-scaled value array (:meth:`_scaled_variables`) -- so JSON and
        CSV stay byte-identical. Each row is formatted with ``csv.writer`` into a
        small per-row buffer so quoting stays identical to a one-shot dump; nothing
        here hand-formats CSV text.
        """
        buffer = io.StringIO()
        writer = csv.writer(buffer, quoting=csv.QUOTE_MINIMAL)

        def _flush(row: list[str]) -> str:
            buffer.seek(0)
            buffer.truncate()
            writer.writerow(row)
            return buffer.getvalue()

        # One row per (date, crossed-zone cell). Each axis describes its own
        # columns (:meth:`Zone.csv_columns`): a structured axis (banded/threshold)
        # expands to two typed, unit-bearing columns, a categorical axis to one.
        # The header comes from a sample cell's columns, every row from its own.
        # Then area + each variable, the shared serialization order.
        # any cell is a faithful per-axis sample: one scheme per axis, >=1 cell always
        sample = self._cells[0]
        headers: list[str] = ['date']
        for layer, zone in zip(self.zone_layers, sample, strict=True):
            headers.extend(header for header, _ in zone.csv_columns(layer))
        headers.append('area_m2')
        headers.extend(self._variable_columns)
        yield _flush(headers)

        cells = self._cells
        emitted = self._emitted_cells(include_empty_zones=include_empty_zones)
        # The zone columns depend only on the cell, not the date; format them once
        # per emitted cell and reuse across every date's row.
        cell_columns = {
            cell_idx: [
                value
                for layer, zone in zip(self.zone_layers, cells[cell_idx], strict=True)
                for _, value in zone.csv_columns(layer)
            ]
            for cell_idx in emitted
        }

        # Scale every value once, then read the shared (area, then variables) order.
        scaled = self._scaled_variables()
        for date_, date_idx in self._dates_index.items():
            for cell_idx in emitted:
                row: list[str] = [
                    date_.isoformat(),
                    *cell_columns[cell_idx],
                    str(float(self._areas[cell_idx])),
                ]
                # Empty cell for a no-data reduction (nan), matching dump_compact's
                # JSON null -- never the literal 'nan'. ``float(...)`` matches the
                # old scalar path's exact text (a Python float repr).
                row.extend(
                    '' if math.isnan(value) else str(float(value))
                    for value in scaled[date_idx, cell_idx]
                )
                yield _flush(row)

    def dump_to_csv(self: Self, out: IO, *, include_empty_zones: bool = False) -> None:
        out.writelines(self.iter_csv(include_empty_zones=include_empty_zones))

    @classmethod
    async def calculate(
        cls: type[Self],
        aoi: AOIRaster,
        rasters: RasterCollection,
        cache: TiffCache,
        dataset: Dataset,
        zones: Sequence[ZoneSelection] = (),
        *,
        registry: Mapping[str, AvailableZone] | None = None,
        max_zone_cells: int = DEFAULT_MAX_ZONE_CELLS,
        max_concurrent_rasters: int = DEFAULT_MAX_CONCURRENT_RASTERS,
    ) -> Self:
        """Reduce ``rasters`` over the AOI, crossed by the selected zone layers.

        ``zones`` is the axes to cross, as already-resolved
        :class:`ZoneSelection`\\ s -- a CLI/HTTP ``LAYER[:PARAM=VALUE]`` string
        token becomes one via :func:`parse_zone_selection` at the shell
        boundary (the reader parses tokens up front, before its
        ``RasterCollection`` build's dataset I/O). An **empty** selection
        means *no* stratification: the reduction is over the whole basin, producing
        a single cell per date whose ``zone`` tuple is empty (the K=0 case of the
        crossed index). Each selected zone layer is read live, windowed to the AOI,
        and assigned to per-pixel ordinals; the crossed index is the cartesian
        product of the axes.

        Axis resolution happens before any raster is read, so an unknown layer
        raises a clean :class:`QueryParameterError`
        up front -- never after paying for I/O. A query whose product would exceed
        ``max_zone_cells`` is likewise rejected before any raster read.

        ``max_concurrent_rasters`` caps how many per-raster reductions run at once
        (a semaphore over the fan-out); it bounds peak memory / fetch fan-out only
        and does not affect results.

        ``registry`` is the zone-layer registry for ``dataset``; the reader already
        built the identical one to parse its tokens and passes it back in to avoid a
        rebuild. When omitted (the programmatic form) it is built here.
        """
        spec = dataset.spec

        # The zone geometry (which pixel is in which crossed cell, and each cell's
        # total area) depends only on the AOI mask + the zone layers -- not on any
        # variable or date -- so the registry it resolves through is built once (or
        # reused from the caller) and shared by every reduction.
        if registry is None:
            registry = available_zones(dataset.providers.values())
        resolved = [resolve_zone_axis(selection, registry, spec) for selection in zones]

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
            return await aoi.read_window(
                dataset.zones[available.provider.name].raster(layer),
                dtype=layer.dtype,
                fill=layer.nodata,
                cache=cache,
            )

        axis_arrays = await asyncio.gather(
            *(_read_axis(available) for available, _ in resolved),
        )
        ordinals_list = [
            scheme.assign(values)
            for (_, scheme), values in zip(resolved, axis_arrays, strict=True)
        ]
        zone_layers = [selection.layer_key for selection in zones]

        zone_index = _ZoneIndex.build(axes, ordinals_list, aoi.array)

        stats = cls(
            rasters.variables,
            tuple(zone_layers),
            zone_index.cell_zones,
            zone_index.areas,
            tuple(rasters.dates),
        )

        # Fan out across the raster set; each raster's tile reads fan out further
        # inside _calc. The handle cache dedupes/bounds open COGs, but each in-flight
        # _calc still holds a transient full-window value array and issues its own
        # (unbounded) fetch batch, so a wide date range would otherwise allocate all
        # of them at once. A semaphore caps how many _calc bodies run concurrently --
        # results are unaffected (each writes its own disjoint (date, variable) slice
        # of the array); only peak memory / fetch fan-out is bounded.
        limit = asyncio.Semaphore(max_concurrent_rasters)

        async def _reduce_one(
            variable: DatasetVariable,
            raster: DataRaster,
        ) -> None:
            async with limit:
                values = await cls._calc(aoi, variable, raster, zone_index, cache)
            stats.fill(raster.date, variable, values)

        await asyncio.gather(
            *(
                _reduce_one(variable, raster)
                for variable, variable_rasters in rasters.items()
                for raster in variable_rasters
            ),
        )

        return stats

    @staticmethod
    async def _calc(
        aoi: AOIRaster,
        variable: DatasetVariable,
        raster: DataRaster,
        zone_index: _ZoneIndex,
        cache: TiffCache,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """The area-weighted per-cell reduction for one (date, variable) raster.

        Returns the cell-vector (aligned with ``zone_index.cell_zones`` flat product
        order) of reduced values -- the value column :meth:`fill` writes for this
        (date, variable). Cell areas are date-invariant and carried on the zone
        index, so they are not recomputed here.
        """
        values_array = await aoi.read_window(
            raster,
            dtype=variable.dtype,
            fill=variable.nodata,
            cache=cache,
        )

        # The reduction runs only over in-zone pixels that actually have data; the
        # zone geometry (which pixel is in which cell, and each cell's area) was
        # precomputed once at build and lives on the index, so _calc hands over only
        # the raw window + its nodata sentinel.
        return zone_index.reduce(variable.reducer, values_array, variable.nodata)


@dataclass
class _ZoneIndex:
    """Per-pixel crossed-zone membership for one AOI, computed once and reused.

    Combines K per-axis ordinal arrays into one **mixed-radix linear index** over
    the product space (size equal to the product of the per-axis zone counts).
    ``in_zone`` is the boolean of pixels that are in every axis' zone *and* in the
    AOI mask; ``areas[c]`` is crossed cell ``c``'s total geographic area.
    ``cell_zones`` carries the per-axis :class:`Zone` tuple for every product cell,
    in the same flat order.

    The reduction never needs the full-window index/area: it works only over the
    in-zone pixels. So the compressed geometry is stored once at build --
    ``idx_c`` is the combined cell index and ``area_c`` the AOI area, each already
    subset to ``in_zone`` (the same subset the ``areas`` bincount consumes) -- and
    :meth:`reduce` compresses only the per-raster value window to match.
    """

    idx_c: numpy.typing.NDArray[numpy.int64]
    area_c: numpy.typing.NDArray[numpy.float32]
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
        # Compress the geometry to the in-zone subset once: the combined cell index
        # and the AOI area over exactly the pixels the reduction (and this areas
        # bincount) will ever touch. Every per-raster reduce fancy-indexes only its
        # value window to this same subset.
        idx_c = combined[in_zone]
        area_c = area[in_zone]
        areas = numpy.bincount(
            idx_c,
            weights=area_c,
            minlength=n,
        ).astype(numpy.float64)
        return cls(
            idx_c=idx_c,
            area_c=area_c,
            in_zone=in_zone,
            areas=areas,
            cell_zones=cls._enumerate_cells(axes),
        )

    @staticmethod
    def _enumerate_cells(
        axes: list[tuple[Zone, ...]],
    ) -> tuple[tuple[Zone, ...], ...]:
        """The product cells in flat (mixed-radix) order: one Zone tuple per cell.

        ``itertools.product`` iterates with the last axis fastest -- the same
        flat order the mixed-radix ``combined`` index in :meth:`build` produces.
        """
        return tuple(itertools.product(*axes))

    def reduce(
        self: Self,
        reducer: Reducer,
        values_array: numpy.typing.NDArray,
        nodata: float,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Area-weighted reduction for every crossed cell at once.

        Runs over the in-zone pixels that also carry data (``!= nodata``). The
        in-zone geometry -- the compressed combined cell index ``idx_c`` and its AOI
        areas ``area_c`` -- was precomputed at :meth:`build`; here only the value
        window is compressed to the in-zone subset, then the nodata mask is applied.
        One pass via ``bincount`` over the combined cell index instead of a per-cell
        masked reduction. Area weighting is automatic from the grid CRS (``area`` is
        geodesic on a geographic grid, constant on a projected one), so MEAN
        degenerates to a plain mean when cells are equal-area. A cell with no
        selected pixels is ``nan`` (as a per-pixel empty reduction would be).
        """
        n = len(self.areas)
        # values_array is over the full window; subset it to the in-zone geometry
        # (matching idx_c/area_c), then drop the nodata pixels within that subset.
        values_c = values_array[self.in_zone]
        has_data = values_c != nodata
        idx = self.idx_c[has_data]
        values = values_c[has_data]
        areas = self.area_c[has_data]
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
            case _:
                # Exhaustiveness: a new Reducer member fails typecheck here
                # instead of surfacing as an UnboundLocalError below.
                assert_never(reducer)

        # Empty cells divide to nan for MEAN already, but TOTAL needs it set
        # explicitly so a no-data cell reads nan rather than a spurious 0.
        result[numpy.bincount(idx, minlength=n) == 0] = numpy.nan
        return result
