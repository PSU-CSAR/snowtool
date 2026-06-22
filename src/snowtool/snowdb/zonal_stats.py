from __future__ import annotations

import asyncio
import csv
import math

from dataclasses import dataclass
from datetime import date
from typing import IO, TYPE_CHECKING, Self, cast

import numpy
import numpy.typing

from snowtool.snowdb.raster import AOIRasterWithArea, DataRaster
from snowtool.snowdb.raster_collection import RasterCollection
from snowtool.snowdb.terrain import ELEVATION, ELEVATION_NODATA
from snowtool.snowdb.variables import DatasetVariable, Reducer
from snowtool.snowdb.zoning import BandZone

if TYPE_CHECKING:
    from pydantic import BaseModel

    from snowtool.snowdb.raster import TiledRaster
    from snowtool.snowdb.spec import DatasetSpec
    from snowtool.snowdb.tiff_cache import TiffCache
    from snowtool.snowdb.zoning import ZoneScheme


@dataclass
class Result:
    date: date
    elevation_band: BandZone
    variable: DatasetVariable
    value: float
    area: float


class ZonalStats:
    def __init__(
        self: Self,
        spec: DatasetSpec,
        variables: set[DatasetVariable],
        elevation_bands: tuple[BandZone, ...],
        dates: tuple[date, ...],
        *results: Result,
    ) -> None:
        self.spec = spec
        self._variables_index = {
            variable: idx + 1
            for idx, variable in enumerate(
                sorted(variables, key=lambda v: v.key),
            )
        }
        # Bands arrive in ordinal (ascending) order from the zone scheme, so the
        # index preserves that order rather than re-sorting.
        self._elevation_bands_index = {
            band: idx for idx, band in enumerate(elevation_bands)
        }
        self._dates_index = {dt: idx for idx, dt in enumerate(sorted(dates))}
        self._array = numpy.full(
            (
                len(self._dates_index),
                len(self._elevation_bands_index),
                len(self._variables_index) + 1,
            ),
            -numpy.inf,
            dtype=numpy.float32,
        )

        self.add_results(*results)

    def add_result(self: Self, result: Result) -> None:
        zone = self._array[self._dates_index[result.date]][
            self._elevation_bands_index[result.elevation_band]
        ]

        zone[0] = result.area
        zone[self._variables_index[result.variable]] = result.value

    def add_results(self: Self, *results: Result) -> None:
        for result in results:
            self.add_result(result)

    def validate(self: Self) -> None:
        if (self._array == -numpy.inf).any():
            raise ValueError(
                'Results array is incomplete. '
                'Ensure all data was processed and added to results successfully.',
            )

    def _zone_stats(self: Self, date_idx: int, band_idx: int) -> dict[str, float]:
        """The scaled per-zone stat values (``area_m2`` + each variable) for one
        (date, band) cell -- the single source the JSON (:meth:`dump`) and CSV
        (:meth:`dump_to_csv`) serializers share, so both apply the same unit
        scaling and ``float`` coercion. The keys are ordered ``area_m2`` first
        then the variables in ``_variables_index`` order, matching the CSV
        header. A band with no valid pixels carries ``nan``, which each
        serializer renders as its own 'missing' token (JSON null / empty cell).
        """
        zone = self._array[date_idx][band_idx]
        values = {'area_m2': float(zone[0])}
        for variable, var_idx in self._variables_index.items():
            values[variable.stat_name] = float(variable.unit.scale(zone[var_idx]))
        return values

    def dump(self: Self) -> list[BaseModel]:
        self.validate()
        stat_model = self.spec.zonal_stat_model
        stats_model = self.spec.zonal_stats_model
        stats: list[BaseModel] = []
        for date_, date_idx in self._dates_index.items():
            zones: list[BaseModel] = []
            for band, band_idx in self._elevation_bands_index.items():
                zones.append(
                    stat_model(
                        min_elevation_ft=band.min,
                        max_elevation_ft=band.max,
                        **self._zone_stats(date_idx, band_idx),
                    ),
                )
            stats.append(stats_model(date=date_, zones=zones))
        return stats

    def dump_to_csv(self: Self, out: IO) -> None:
        self.validate()
        writer = csv.writer(out, quoting=csv.QUOTE_MINIMAL)

        headers: list[str] = ['date']
        for band in self._elevation_bands_index:
            headers.append(f'area_m2_{band}')
            for variable in self._variables_index:
                headers.append(f'{variable.stat_name}_{band}')

        writer.writerow(headers)

        for date_, date_idx in self._dates_index.items():
            row: list[str] = [date_.isoformat()]
            for band_idx in self._elevation_bands_index.values():
                # Empty cell for a no-data band (nan), matching dump()'s JSON
                # null -- never the literal 'nan'.
                row.extend(
                    '' if math.isnan(value) else str(value)
                    for value in self._zone_stats(date_idx, band_idx).values()
                )
            writer.writerow(row)

    @classmethod
    async def calculate(
        cls: type[Self],
        aoi: AOIRasterWithArea,
        rasters: RasterCollection,
        cache: TiffCache,
        spec: DatasetSpec,
        elevation: TiledRaster[numpy.float32],
    ) -> Self:
        # Elevation bands come from the elevation layer's banded zoning scheme,
        # stepped by this dataset's band_step_ft. The scheme spans the global
        # bracket (shared by every dataset), so a given band means the same thing
        # across AOIs and datasets; the per-AOI band geometry below restricts which
        # of them actually carry data. (ELEVATION always declares a scheme.)
        scheme = ELEVATION.zoning
        assert scheme is not None  # noqa: S101 - ELEVATION always has a zoning scheme

        # Elevation is read live from the dataset's terrain set (the AOI raster is
        # a bare geometry mask now), windowed to the AOI's tiles.
        elevation_array = numpy.full(
            aoi.array.shape,
            ELEVATION_NODATA,
            dtype=numpy.float32,
        )
        await aoi.load_raster_tiles_into_array(elevation, elevation_array, cache)

        # The band geometry (which pixel is in which band, and each band's total
        # area) depends only on the AOI mask + elevation -- not on any variable or
        # date -- so it is computed once here and reused by every reduction.
        band_index = _BandIndex.build(
            scheme,
            elevation_array,
            aoi.array,
            aoi.area,
            step=spec.band_step_ft,
        )

        # Fan out across the raster set; each raster's tile reads fan out
        # further inside _calc. The handle cache dedupes/bounds open COGs.
        per_raster = await asyncio.gather(
            *(
                cls._calc(aoi, variable, raster, band_index, cache)
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
            band_index.bands,
            tuple(rasters.dates),
            *results,
        )

    @staticmethod
    async def _calc(
        aoi: AOIRasterWithArea,
        variable: DatasetVariable,
        raster: DataRaster,
        band_index: _BandIndex,
        cache: TiffCache,
    ) -> list[Result]:
        date_ = raster.date
        values_array = numpy.empty_like(aoi.array, dtype=variable.dtype)
        values_array[:] = variable.nodata

        await aoi.load_raster_tiles_into_array(raster, values_array, cache)

        # The reduction runs only over in-band pixels that actually have data;
        # everything else (band geometry, band areas) was precomputed once.
        selection = band_index.in_band & (values_array != variable.nodata)
        values = band_index.reduce(variable.reducer, values_array, aoi.area, selection)

        return [
            Result(
                date=date_,
                variable=variable,
                elevation_band=band,
                value=float(values[idx]),
                area=float(band_index.areas[idx]),
            )
            for idx, band in enumerate(band_index.bands)
        ]


@dataclass
class _BandIndex:
    """Per-pixel zone membership for one AOI along one banded axis, reused.

    ``index`` maps every pixel to its zone ordinal (``-1`` for pixels out of
    zone -- below/above the domain or layer-nodata); ``in_band`` is the boolean of
    in-zone, in-mask pixels; ``areas[b]`` is band ``b``'s total geographic area.
    Reused across every (variable, date) raster so the masks and areas are not
    recomputed per raster -- only each raster's data-dependent reduction is.
    """

    bands: tuple[BandZone, ...]
    index: numpy.typing.NDArray[numpy.int64]
    in_band: numpy.typing.NDArray[numpy.bool_]
    areas: numpy.typing.NDArray[numpy.float64]

    @classmethod
    def build(
        cls: type[Self],
        scheme: ZoneScheme,
        values: numpy.typing.NDArray[numpy.float32],
        mask: numpy.typing.NDArray[numpy.uint8],
        area: numpy.typing.NDArray[numpy.float32],
        *,
        step: int | None = None,
    ) -> Self:
        """Build the index from a banded zoning scheme over ``values``.

        ``scheme.assign`` maps each pixel to its band ordinal (``-1`` out of zone),
        so the scheme owns the unit scaling and nodata handling; this just ANDs the
        AOI mask in and tallies per-band area.
        """
        # The single-axis index is banded (elevation), so the scheme's zones are
        # BandZones; the cast carries that to the typed Result.
        bands = cast('tuple[BandZone, ...]', scheme.zones(step=step))
        n = len(bands)
        index = scheme.assign(values, step=step)
        in_band = (index >= 0) & (mask != 0)
        areas = numpy.bincount(
            index[in_band],
            weights=area[in_band],
            minlength=n,
        ).astype(numpy.float64)
        return cls(bands=bands, index=index, in_band=in_band, areas=areas)

    def reduce(
        self: Self,
        reducer: Reducer,
        values_array: numpy.typing.NDArray,
        area_array: numpy.typing.NDArray[numpy.float32],
        selection: numpy.typing.NDArray[numpy.bool_],
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Area-weighted reduction for every band at once, over ``selection``.

        One pass via ``bincount`` instead of a per-band masked reduction. Area
        weighting is automatic from the grid CRS (``area`` is geodesic on a
        geographic grid, constant on a projected one), so MEAN degenerates to a
        plain mean when cells are equal-area. A band with no selected pixels is
        ``nan`` (as a per-pixel empty reduction would be).
        """
        n = len(self.bands)
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

        # Empty bands divide to nan for MEAN already, but TOTAL needs it set
        # explicitly so a no-data band reads nan rather than a spurious 0.
        result[numpy.bincount(idx, minlength=n) == 0] = numpy.nan
        return result
