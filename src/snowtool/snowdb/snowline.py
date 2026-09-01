from datetime import date
from itertools import pairwise

from snowtool.exceptions import QueryParameterError, SnowLineError
from snowtool.snowdb.snowline_models import SnowLine
from snowtool.snowdb.zonal_stat_models import BandZoneRef, CompactStats


def _center(band: BandZoneRef) -> float:
    return (band.min + band.max) / 2


def _variable_index(stats: CompactStats, variable: str | None) -> int:
    if variable is None:
        if len(stats.variables) != 1:
            raise SnowLineError(
                'stats contain multiple variables; please specify which to use: '
                f'{stats.variables}',
            )
        return 0
    try:
        return stats.variables.index(variable)
    except ValueError:
        raise QueryParameterError(
            f'Variable {variable} was not found in stats: '
            f'Available Stats: {stats.variables}',
        ) from None


def _elevation_bands(stats: CompactStats) -> list[tuple[int, BandZoneRef]]:

    if stats.zone_layers != ['terrain.elevation']:
        raise SnowLineError(
            f'Snow Line Interpolation requires a single terrain.elevation zone axis; '
            f'got {stats.zone_layers}',
        )

    bands: list[tuple[int, BandZoneRef]] = []
    for index, cell in enumerate(stats.zones):
        if len(cell.zone) != 1:
            raise SnowLineError(
                f'Expected one zone reference per cell, got {len(cell.zone)}',
            )
        ref = cell.zone[0]
        if not isinstance(ref, BandZoneRef):
            raise SnowLineError(
                f'Expected a banded zone axis, got {ref.kind!r}',
            )
        bands.append((index, ref))

    if len(bands) < 2:
        raise SnowLineError(
            'Snow line interpolation requires at least two elevations '
            f'; only got {len(bands)}',
        )

    bands.sort(key=lambda pair: pair[1].min)
    return bands


def snow_line_elevation(
    stats: CompactStats,
    *,
    threshold: float = 50.0,
    variable: str | None = None,
) -> dict[date, SnowLine | None]:

    if threshold <= 0.0:
        raise SnowLineError(
            'Invalid Snow line threshold passed: '
            f'Expecting positive value to interpolate: got {threshold}',
        )

    bands = _elevation_bands(stats)
    var = _variable_index(stats, variable)

    resp: dict[date, SnowLine | None] = {}
    for d, values in stats.results.items():
        resp[d] = None

        for (lo_i, lo_band), (hi_i, hi_band) in pairwise(bands):
            lo_val = values[lo_i][var]
            hi_val = values[hi_i][var]
            if lo_val is None or hi_val is None:
                continue
            if lo_val < threshold <= hi_val:
                frac = (threshold - lo_val) / (hi_val - lo_val)
                span = _center(hi_band) - _center(lo_band)
                resp[d] = SnowLine(
                    elevation_ft=(_center(lo_band) + frac * (span)),
                    lower_band_area_m2=stats.zones[lo_i].area_m2,
                    higher_band_area_m2=stats.zones[hi_i].area_m2,
                    interpolation_span_ft=span,
                )
                break

    return resp
