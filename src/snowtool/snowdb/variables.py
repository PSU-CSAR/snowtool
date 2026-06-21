"""Dataset variables: the per-variable definition used by reads and stats.

A :class:`DatasetVariable` lives on a :class:`~snowtool.snowdb.spec.DatasetSpec`
and says, for one requestable variable: how to find its files in a date dir
(``glob``), how to read them (``dtype``/``nodata``), how to reduce per elevation
band (``reducer``), and how to report its value (``unit``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Self


class Reducer(StrEnum):
    MEAN = 'mean'  # area-weighted average over valid pixels
    INTEGRAL = 'integral'  # area-weighted sum over valid pixels -- a basin total


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

    @property
    def stat_name(self: Self) -> str:
        """The reduced-stat field name, e.g. ``mean_swe_mm``."""
        return f'{self.reducer.value}_{self.key}_{self.unit.name}'
