import asyncio
import csv

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import IO, TYPE_CHECKING, Self

import numpy
import numpy.typing

from snowtool import types
from snowtool.snowdb.elevation_band import ElevationBand
from snowtool.snowdb.raster import AOIRasterWithArea, DataRaster
from snowtool.snowdb.raster_collection import RasterCollection
from snowtool.snowdb.variables import DatasetVariable, Reducer

if TYPE_CHECKING:
    from snowtool.snowdb.spec import DatasetSpec
    from snowtool.snowdb.tiff_cache import TiffCache


@dataclass
class Result:
    date: date
    elevation_band: ElevationBand
    variable: DatasetVariable
    value: float
    area: float


class ZonalStats:
    def __init__(
        self: Self,
        variables: set[DatasetVariable],
        elevation_bands: tuple[ElevationBand, ...],
        dates: tuple[date, ...],
        *results: Result,
    ) -> None:
        self._variables_index = {
            variable: idx + 1
            for idx, variable in enumerate(
                sorted(variables, key=lambda v: v.key),
            )
        }
        self._elevation_bands_index = {
            band: idx for idx, band in enumerate(sorted(elevation_bands))
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

    def dump(self: Self) -> list[types.SnodasZonalStats]:
        self.validate()
        stats: list[types.SnodasZonalStats] = []
        for date_, date_idx in self._dates_index.items():
            zones: list[types.SnodasZonalStat] = []
            for band, band_idx in self._elevation_bands_index.items():
                results = {
                    'area_m2': self._array[date_idx][band_idx][0],
                }
                for variable, var_idx in self._variables_index.items():
                    results[variable.stat_name] = variable.unit.scale(
                        self._array[date_idx][band_idx][var_idx],
                    )
                zones.append(
                    types.SnodasZonalStat(
                        min_elevation_ft=band.min,
                        max_elevation_ft=band.max,
                        **results,
                    ),
                )
            stats.append(
                types.SnodasZonalStats(
                    date=date_,
                    zones=zones,
                ),
            )
        return stats

    def dump_to_csv(self: Self, out: IO) -> None:
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
                row.append(
                    self._array[date_idx][band_idx][0],
                )
                for variable, var_idx in self._variables_index.items():
                    row.append(
                        str(
                            variable.unit.scale(
                                self._array[date_idx][band_idx][var_idx],
                            ),
                        ),
                    )
            writer.writerow(row)

    @classmethod
    async def calculate(
        cls: type[Self],
        aoi: AOIRasterWithArea,
        rasters: RasterCollection,
        cache: TiffCache,
        spec: DatasetSpec,
    ) -> Self:
        elevation_bands = tuple(
            ElevationBand.generate(
                size_ft=spec.band_step_ft,
                min_elevation=spec.dem_min_m,
                max_elevation=spec.dem_max_m,
            ),
        )

        # Fan out across the raster set; each raster's tile reads fan out
        # further inside _calc. The handle cache dedupes/bounds open COGs.
        per_raster = await asyncio.gather(
            *(
                cls._calc(aoi, variable, raster, elevation_bands, cache)
                for variable, variable_rasters in rasters.items()
                for raster in variable_rasters
            ),
        )
        results: list[Result] = [
            result for raster_results in per_raster for result in raster_results
        ]

        return cls(
            rasters.variables,
            elevation_bands,
            tuple(rasters.dates),
            *results,
        )

    @staticmethod
    async def _calc(
        aoi: AOIRasterWithArea,
        variable: DatasetVariable,
        raster: DataRaster,
        elevation_bands: Iterable[ElevationBand],
        cache: TiffCache,
    ) -> list[Result]:
        date_ = raster.date
        values_array = numpy.empty_like(aoi.array, dtype=variable.dtype)
        values_array[:] = variable.nodata

        await aoi.load_raster_tiles_into_array(raster, values_array, cache)

        results: list[Result] = []
        for band in elevation_bands:
            # The band's geographic area is variable-independent: every pixel
            # whose elevation falls in the band, regardless of whether this
            # variable has data there.
            band_selection: numpy.typing.NDArray[numpy.bool_] = (
                aoi.array >= band.min_meters
            ) & (aoi.array < band.max_meters)
            # The reduction runs only over pixels that actually have data.
            value_selection = band_selection & (values_array != variable.nodata)

            area = float(numpy.sum(aoi.area[band_selection]))
            value = _reduce(
                variable.reducer,
                values_array[value_selection],
                aoi.area[value_selection],
            )

            results.append(
                Result(
                    date=date_,
                    variable=variable,
                    elevation_band=band,
                    value=value,
                    area=area,
                ),
            )

        return results


def _reduce(
    reducer: Reducer,
    values: numpy.typing.NDArray,
    areas: numpy.typing.NDArray,
) -> float:
    """Reduce the selected (value, per-pixel area) pairs to a single number.

    Area weighting is automatic from the grid CRS (``areas`` is geodesic on a
    geographic grid, constant on a projected one), so MEAN degenerates to a plain
    mean when cells are equal-area. Returns ``nan`` when no pixels are selected.
    """
    if values.size == 0:
        return numpy.nan
    match reducer:
        case Reducer.MEAN:
            return float(numpy.average(values, weights=areas))
        case Reducer.INTEGRAL:
            return float(numpy.sum(values * areas))
