"""Dataset variables: the per-variable definition used by reads and stats.

A :class:`DatasetVariable` lives on a :class:`~snowtool.snowdb.spec.DatasetSpec`
and says, for one requestable variable: how to find its files in a date dir
(``glob``), how to read them (``dtype``/``nodata``), how to reduce per elevation
band (``reducer``), and how to report its value (``unit``).
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from enum import StrEnum
from typing import Self


class Reducer(StrEnum):
    MEAN = 'mean'  # area-weighted average over valid pixels
    # area-weighted accumulation Sum(value*area) over valid pixels -- a basin
    # total (an extensive whole-basin quantity, e.g. a volume), NOT a bare
    # Sum(value): the area weighting is what makes it meaningful.
    TOTAL = 'total'


@dataclass(frozen=True)
class Unit:
    name: str
    scale_factor: float

    def scale(self: Self, value: float) -> float:
        return value / self.scale_factor


@dataclass(frozen=True)
class DatasetVariable:
    key: str
    unit: Unit
    reducer: Reducer
    dtype: str  # numpy read dtype, e.g. 'int16', 'float32'
    nodata: float
    glob: str  # filename glob within a cogs/<YYYYMMDD>/ dir

    def __post_init__(self: Self) -> None:
        # COGs are named `<source-provenance>__<key>.tif` and the read glob
        # anchors on that `__` boundary (`*__swe.tif`). A `__` inside a key would
        # let one variable's glob match another's file, so keys must not contain
        # it. (Single underscores are fine; the delimiter is doubled to stay
        # distinct from in-key separators like `viewable_snow_fraction`.)
        if '__' in self.key:
            raise ValueError(
                f'DatasetVariable key {self.key!r} must not contain "__": it is '
                'the COG filename delimiter separating provenance from variable.',
            )
        # ``nodata`` must be a finite sentinel. The stats reader excludes fill
        # pixels with ``values != variable.nodata`` (zonal_stats.py), and IEEE
        # ``x != NaN`` is ``True`` for every ``x`` -- so a NaN sentinel would let
        # every fill pixel through and NaN-poison the reduction rather than being
        # masked out. This is the same finite-sentinel requirement the terrain
        # elevation/entropy layers rely on (zones/terrain.py) so that fill values
        # digitise cleanly out of a zoning scheme.
        if math.isnan(self.nodata):
            raise ValueError(
                f'DatasetVariable {self.key!r} nodata must be a finite sentinel, '
                'not NaN: the stats reader masks fill pixels with a `!=` compare, '
                'which can never exclude NaN (`x != NaN` is always True), so a NaN '
                'fill would poison every reduction. Use a finite out-of-range value.',
            )

    @property
    def stat_name(self: Self) -> str:
        """The reduced-stat field name, e.g. ``mean_swe_mm``."""
        return f'{self.reducer.value}_{self.key}_{self.unit.name}'
