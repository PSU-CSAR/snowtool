"""The zone model: how a layer's pixels map to zones.

A :class:`ZoneScheme` declares the zones a
:class:`~snowtool.snowdb.zones.zone_layer.ZoneLayer` stratifies the grid into and how
each pixel is assigned to one. Three kinds are supported:

* :class:`BandedZoning` -- contiguous numeric bands aligned to 0 over a fixed
  domain (elevation in feet, forest cover in percent).
* :class:`ThresholdZoning` -- a binary below/at-or-above split (forest cover
  forested vs unforested).
* :class:`CategoricalZoning` -- a fixed set of discrete classes (aspect
  N/E/S/W/flat).

Both map every pixel to a per-pixel zone *ordinal* via :meth:`ZoneScheme.assign`,
where ``-1`` means "out of zone" -- a single value that uniformly covers
layer-nodata and out-of-domain pixels, so the zonal-stats engine excludes them
the same way regardless of scheme.

The per-axis :class:`Zone` descriptors (:class:`BandZone`, :class:`ClassZone`)
name the zones a scheme produces; they are the self-describing cells a crossed
query reports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy
import numpy.typing

from snowtool.exceptions import QueryParameterError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class Zone:
    """A single zone along one axis -- one cell of a :class:`ZoneScheme`.

    The base carries the identity every zone shares: ``key`` is a stable id for
    the zone within its axis (a band's ``'<min>_<max>'`` or a class's name) and
    ``label`` is its human label.
    """

    key: str
    label: str

    def ref_fields(self: Self) -> dict[str, object]:
        """The response-model fields for this zone (incl. its ``kind`` tag).

        Merged with the axis ``layer`` key and validated into the matching
        :class:`~snowtool.snowdb.zonal_stat_models.ZoneRef` member, so each kind
        owns its own self-description rather than being switched on externally.
        """
        raise NotImplementedError

    def csv_columns(self: Self, layer: str) -> list[tuple[str, str]]:
        """This axis' ``(header, value)`` CSV column pairs for one crossed cell.

        The header names the column (qualified by ``layer`` and the zone unit);
        the value is this zone's entry. A structured axis (banded/threshold)
        expands to two columns, a categorical axis to one.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class BandZone(Zone):
    """A contiguous numeric band ``[min, max)`` in the scheme's zone ``unit``.

    ``min``/``max`` are whole zone-unit bounds (feet for elevation, percent for
    forest cover).
    """

    min: int
    max: int
    unit: str

    def __str__(self: Self) -> str:
        return f'{self.min}_{self.max}'

    def ref_fields(self: Self) -> dict[str, object]:
        return {'kind': 'band', 'min': self.min, 'max': self.max, 'unit': self.unit}

    def csv_columns(self: Self, layer: str) -> list[tuple[str, str]]:
        return [
            (f'{layer}_min_{self.unit}', str(self.min)),
            (f'{layer}_max_{self.unit}', str(self.max)),
        ]


@dataclass(frozen=True)
class ClassZone(Zone):
    """One discrete class, identified by its on-disk pixel ``code``."""

    code: int

    def __str__(self: Self) -> str:
        return self.label

    def ref_fields(self: Self) -> dict[str, object]:
        return {'kind': 'class', 'code': self.code, 'label': self.label}

    def csv_columns(self: Self, layer: str) -> list[tuple[str, str]]:
        return [(layer, self.label)]


@dataclass(frozen=True)
class ThresholdZone(Zone):
    """One side of a threshold split, carrying the structured split point.

    ``side`` is ``'below'`` or ``'above'`` (at-or-above); ``threshold`` (in
    ``unit``) is the split point, exposed as a real value rather than buried in the
    label, so a consumer can read/filter on it.
    """

    threshold: float
    unit: str
    side: str

    def __str__(self: Self) -> str:
        return self.label

    def ref_fields(self: Self) -> dict[str, object]:
        return {
            'kind': 'threshold',
            'threshold': self.threshold,
            'unit': self.unit,
            'side': self.side,
            'label': self.label,
        }

    def csv_columns(self: Self, layer: str) -> list[tuple[str, str]]:
        return [
            (f'{layer}_side', self.label),
            (f'{layer}_threshold_{self.unit}', f'{self.threshold:g}'),
        ]


class ZoneScheme(ABC):
    """How one zone layer's pixels map to zones.

    :meth:`zones` enumerates the scheme's zones in ordinal order (optionally
    overriding its defaults), and :meth:`assign` maps an array of native pixel
    values to per-pixel zone ordinals (``-1`` = out of zone, which uniformly
    covers layer-nodata and out-of-domain values).
    """

    @abstractmethod
    def zones(self: Self, **override: object) -> tuple[Zone, ...]:
        """The scheme's zones, in ordinal order."""
        raise NotImplementedError

    @abstractmethod
    def assign(
        self: Self,
        values: numpy.typing.NDArray,
        **override: object,
    ) -> numpy.typing.NDArray[numpy.int64]:
        """Per-pixel zone ordinal for ``values`` (``-1`` where out of zone)."""
        raise NotImplementedError

    def default_overrides(
        self: Self,
        params: Mapping[str, object],
    ) -> dict[str, object]:
        """Scheme override kwargs from a dataset's configured zone ``params``.

        Translates the human-facing param a dataset's ``zones`` block carries (e.g.
        ``band_step_ft``, ``threshold_pct``) into the kwarg :meth:`zones`/
        :meth:`assign` consume (``step``/``threshold``). The base scheme takes no
        params (so a categorical layer ignores any), so it returns nothing.
        """
        return {}

    def parse_override(self: Self, layer_key: str, raw: str) -> int | float:
        """Parse a query's ``:override`` token (the CLI ``--zone`` flag) for this
        scheme. The base scheme takes none, so any token is an error -- a
        categorical axis has nothing to override."""
        raise QueryParameterError(
            f'zone {layer_key!r} takes no override (it is a categorical axis); '
            f'drop the ":{raw}".',
        )

    def override_kwargs(self: Self, override: int | float) -> dict[str, object]:
        """The scheme kwargs an explicit query override resolves to.

        The counterpart of :meth:`default_overrides` for the *explicit* override a
        selection carries (vs. the dataset's configured default). The base scheme
        consumes no override, so it returns nothing.
        """
        return {}


@dataclass(frozen=True)
class BandedZoning(ZoneScheme):
    """Contiguous numeric bands aligned to 0 over ``[domain_min, domain_max]``.

    The domain is expressed in *zone* units (e.g. feet for elevation, percent for
    forest cover); ``value_scale`` maps native pixel units to those zone units
    (elevation pixels are metres, so ``value_scale`` is ``M_TO_FT``; forest pixels
    are already percent, so it is ``1``). Bands are aligned to 0 so a given band
    means the same thing regardless of the domain, and ``default_step`` is the band
    width used when a query does not override it (e.g. elevation's per-dataset
    ``band_step_ft``).
    """

    domain_min: float
    domain_max: float
    default_step: int
    unit: str
    value_scale: float
    layer_nodata: float
    # The key a dataset's zones block uses for this layer's band width (e.g.
    # ``band_step_ft``); its value becomes the ``step`` override.
    param_key: str = 'band_step_ft'

    def default_overrides(
        self: Self,
        params: Mapping[str, object],
    ) -> dict[str, object]:
        if self.param_key in params:
            return {'step': params[self.param_key]}
        return {}

    def parse_override(self: Self, layer_key: str, raw: str) -> int:
        try:
            return int(raw)
        except ValueError as e:
            raise QueryParameterError(
                f'zone {layer_key!r} band step must be an integer, got {raw!r}.',
            ) from e

    def override_kwargs(self: Self, override: int | float) -> dict[str, object]:
        return {'step': override}

    def _step(self: Self, override: object) -> int:
        step = override if override is not None else self.default_step
        if not isinstance(step, int) or step <= 0:
            raise ValueError(f'band step must be a positive int, got {step!r}')
        return step

    def zones(self: Self, **override: object) -> tuple[BandZone, ...]:
        """The bands spanning the domain at ``step`` (default ``default_step``).

        Aligned to 0: band ``i`` is ``[i*step, (i+1)*step)``. Reproduces the old
        ``ElevationBand.generate`` for the elevation domain.
        """
        step = self._step(override.get('step'))
        start = int(self.domain_min // step)
        end = int(self.domain_max // step) + 1
        return tuple(
            BandZone(
                key=f'{i * step}_{(i + 1) * step}',
                label=f'{i * step}-{(i + 1) * step} {self.unit}',
                min=i * step,
                max=(i + 1) * step,
                unit=self.unit,
            )
            for i in range(start, end)
        )

    def assign(
        self: Self,
        values: numpy.typing.NDArray,
        **override: object,
    ) -> numpy.typing.NDArray[numpy.int64]:
        """Digitize ``values`` (native units) into the band ordinals.

        Scales to zone units and digitizes into the band edges; pixels below/above
        the domain, and layer-nodata pixels, become ``-1``.
        """
        bands = self.zones(**override)
        edges = numpy.array(
            [band.min for band in bands] + [bands[-1].max],
            dtype=numpy.float64,
        )
        scaled = numpy.asarray(values, dtype=numpy.float64) * self.value_scale
        ordinals = (numpy.digitize(scaled, edges) - 1).astype(numpy.int64)
        ordinals[(ordinals < 0) | (ordinals >= len(bands))] = -1
        # Belt-and-suspenders: an explicit nodata sentinel is excluded even if it
        # somehow scaled into the domain.
        ordinals[numpy.asarray(values) == self.layer_nodata] = -1
        return ordinals


@dataclass(frozen=True)
class ThresholdZoning(ZoneScheme):
    """A binary split at a threshold: *below* vs *at-or-above* it.

    The query unit (e.g. forest cover: "below 50% is unforested, 50%+ is
    forested"). ``value_scale`` maps native pixel units to the threshold's unit;
    ``default_threshold`` is used when a query does not override it. The two zones
    are :class:`ClassZone`\\ s (ordinal 0 below, 1 at-or-above) whose labels embed
    the active threshold so each cell stays self-describing.
    """

    default_threshold: float
    unit: str
    value_scale: float
    layer_nodata: float
    below_label: str
    above_label: str
    # The key a dataset's zones block uses for this layer's split point (e.g.
    # ``threshold_pct``); its value becomes the ``threshold`` override.
    param_key: str = 'threshold_pct'

    def default_overrides(
        self: Self,
        params: Mapping[str, object],
    ) -> dict[str, object]:
        if self.param_key in params:
            return {'threshold': params[self.param_key]}
        return {}

    def parse_override(self: Self, layer_key: str, raw: str) -> float:
        try:
            return float(raw)
        except ValueError as e:
            raise QueryParameterError(
                f'zone {layer_key!r} threshold must be a number, got {raw!r}.',
            ) from e

    def override_kwargs(self: Self, override: int | float) -> dict[str, object]:
        return {'threshold': override}

    def _threshold(self: Self, override: object) -> float:
        if override is not None and not isinstance(override, int | float):
            raise ValueError(f'threshold must be a number, got {override!r}')
        return float(self.default_threshold if override is None else override)

    def zones(self: Self, **override: object) -> tuple[ThresholdZone, ...]:
        """The two sides of the split (below, at-or-above), with clean labels.

        The active threshold rides on each :class:`ThresholdZone` as a structured
        value (not embedded in the label).
        """
        threshold = self._threshold(override.get('threshold'))
        return (
            ThresholdZone(
                key='below',
                label=self.below_label,
                threshold=threshold,
                unit=self.unit,
                side='below',
            ),
            ThresholdZone(
                key='above',
                label=self.above_label,
                threshold=threshold,
                unit=self.unit,
                side='above',
            ),
        )

    def assign(
        self: Self,
        values: numpy.typing.NDArray,
        **override: object,
    ) -> numpy.typing.NDArray[numpy.int64]:
        """1 where ``values`` (scaled) >= threshold, 0 below, ``-1`` for nodata."""
        threshold = self._threshold(override.get('threshold'))
        scaled = numpy.asarray(values, dtype=numpy.float64) * self.value_scale
        out = (scaled >= threshold).astype(numpy.int64)
        out[numpy.asarray(values) == self.layer_nodata] = -1
        return out


@dataclass(frozen=True)
class CategoricalZoning(ZoneScheme):
    """A fixed set of discrete classes keyed by their on-disk pixel codes."""

    classes: tuple[ClassZone, ...]
    layer_nodata: int

    def zones(self: Self, **override: object) -> tuple[ClassZone, ...]:
        """The class list (its order *is* the ordinal order)."""
        return self.classes

    def assign(
        self: Self,
        values: numpy.typing.NDArray,
        **override: object,
    ) -> numpy.typing.NDArray[numpy.int64]:
        """Map each pixel's class code to its ordinal (``-1`` for nodata/unknown).

        Layer-nodata and any code not in :attr:`classes` fall through to ``-1``.
        """
        out = numpy.full(numpy.shape(values), -1, dtype=numpy.int64)
        for ordinal, cls in enumerate(self.classes):
            out[numpy.asarray(values) == cls.code] = ordinal
        return out


def banded(
    *,
    domain_min: float,
    domain_max: float,
    default_step: int,
    unit: str,
    value_scale: float,
    layer_nodata: float,
    param_key: str = 'band_step_ft',
) -> BandedZoning:
    """Convenience constructor for :class:`BandedZoning` (keyword-only)."""
    return BandedZoning(
        domain_min=domain_min,
        domain_max=domain_max,
        default_step=default_step,
        unit=unit,
        value_scale=value_scale,
        layer_nodata=layer_nodata,
        param_key=param_key,
    )


def categorical(
    classes: Sequence[ClassZone],
    *,
    layer_nodata: int,
) -> CategoricalZoning:
    """Convenience constructor for :class:`CategoricalZoning`."""
    return CategoricalZoning(classes=tuple(classes), layer_nodata=layer_nodata)


def threshold(
    *,
    default_threshold: float,
    unit: str,
    value_scale: float,
    layer_nodata: float,
    below_label: str,
    above_label: str,
    param_key: str = 'threshold_pct',
) -> ThresholdZoning:
    """Convenience constructor for :class:`ThresholdZoning` (keyword-only)."""
    return ThresholdZoning(
        default_threshold=default_threshold,
        unit=unit,
        value_scale=value_scale,
        layer_nodata=layer_nodata,
        below_label=below_label,
        above_label=above_label,
        param_key=param_key,
    )
