"""Dataset variables: the per-variable definition used by reads and stats.

A :class:`DatasetVariable` lives on a :class:`~snowtool.snowdb.spec.DatasetSpec`
and says, for one requestable variable: how to find its files in a date dir
(``glob``), how to read them (``dtype``/``nodata``), how to reduce per elevation
band (``reducer``), and how to report its value (``unit``).
"""

from __future__ import annotations

import math

from enum import StrEnum
from typing import Self

import numpy

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Reducer(StrEnum):
    MEAN = 'mean'  # area-weighted average over valid pixels
    # area-weighted accumulation Sum(value*area) over valid pixels -- a basin
    # total (an extensive whole-basin quantity, e.g. a volume), NOT a bare
    # Sum(value): the area weighting is what makes it meaningful.
    TOTAL = 'total'


class Unit(BaseModel):
    """A variable's reporting unit, serialized inline as ``{name, scale_factor}``.

    Frozen (so it is hashable and can ride on the hashable
    :class:`DatasetVariable`) and its own persisted form -- there is no separate
    config mirror.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    scale_factor: float

    def scale(self: Self, value: float) -> float:
        return value / self.scale_factor


class DatasetVariable(BaseModel):
    """One requestable variable: how to find, read, reduce, and report it.

    Frozen (hashable): used in sets and as dict keys in the zonal-stats engine.
    It is also its own persisted form -- a dataset config stores its variables as
    ``{key: {unit, reducer, dtype, nodata, glob}}`` with the ``key`` supplied by
    the dict key, so ``key`` is excluded from serialization here and injected by
    :class:`~snowtool.snowdb.config.DatasetConfig` on load.
    """

    model_config = ConfigDict(frozen=True)

    # The dict key of a config's ``variables`` map *is* this key, so it is not
    # written back inside the value (Field(exclude=True)); DatasetConfig injects
    # it from the map key on load. It still participates in equality and hashing.
    key: str = Field(exclude=True)
    unit: Unit
    reducer: Reducer
    dtype: str  # numpy read dtype, e.g. 'int16', 'float32'
    nodata: float
    glob: str  # filename glob within a cogs/<YYYYMMDD>/ dir

    @field_validator('dtype')
    @classmethod
    def _dtype_parses(cls: type[Self], value: str) -> str:
        """Reject a dtype numpy cannot parse at config load, not first read."""
        try:
            numpy.dtype(value)
        except TypeError as e:
            raise ValueError(f'dtype {value!r} is not a numpy dtype') from e
        return value

    @model_validator(mode='after')
    def _validate(self: Self) -> Self:
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
        return self

    @property
    def stat_name(self: Self) -> str:
        """The reduced-stat field name, e.g. ``mean_swe_mm``."""
        return f'{self.reducer.value}_{self.key}_{self.unit.name}'
