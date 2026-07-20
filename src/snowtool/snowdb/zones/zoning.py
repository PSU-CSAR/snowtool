"""The zone model: how a layer's pixels map to zones.

A :class:`ZoneScheme` declares the zones a
:class:`~snowtool.snowdb.zones.zone_layer.ZoneLayer` stratifies the grid into and how
each pixel is assigned to one. Three kinds are supported:

* :class:`BandedZoning` -- contiguous numeric bands of a fixed *width* aligned to 0
  over a fixed domain (elevation in feet).
* :class:`EvenBucketZoning` -- a fixed *count* of equal buckets over a closed domain,
  for a bounded dimensionless measure (the [-1, 1] aspect components).
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
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Literal, Self

import numpy
import numpy.typing

from snowtool.exceptions import QueryParameterError, ZoneParamsError
from snowtool.snowdb.config import (
    BandStepParams,
    BucketParams,
    EntropyThresholdParams,
    ThresholdParams,
)
from snowtool.snowdb.zonal_stat_models import (
    BandZoneRef,
    ClassZoneRef,
    ThresholdZoneRef,
)

if TYPE_CHECKING:
    from snowtool.snowdb.config import ZoneLayerParams
    from snowtool.snowdb.zonal_stat_models import ZoneRef


@dataclass(frozen=True)
class ZoneClassDescription:
    """One categorical class in a scheme's :class:`ZoneDescription` (key + label)."""

    key: str
    label: str


@dataclass(frozen=True)
class BandedZoneDescription:
    """Self-description of a banded axis: its override param, band width
    default, unit, and covered range (first band's lower to last band's upper
    edge). The switch-free surface the API discovery/query layers read instead
    of ``isinstance``-ing a scheme (likewise the other kinds below)."""

    kind: ClassVar[Literal['banded']] = 'banded'
    param_key: str
    default: int
    unit: str
    min: int | float
    max: int | float


@dataclass(frozen=True)
class BucketedZoneDescription:
    """Self-description of an even-bucketed axis (dimensionless: no unit)."""

    kind: ClassVar[Literal['bucketed']] = 'bucketed'
    param_key: str
    default: int
    min: int | float
    max: int | float
    # Kept as an attribute so the overridable kinds share a uniform shape for
    # consumers reading ``desc.unit``; a bucketed axis is always dimensionless.
    unit: ClassVar[None] = None


@dataclass(frozen=True)
class ThresholdZoneDescription:
    """Self-description of a threshold-split axis: param, split default, unit,
    and the measured range the split sits within."""

    kind: ClassVar[Literal['threshold']] = 'threshold'
    param_key: str
    default: float
    unit: str
    min: int | float
    max: int | float


@dataclass(frozen=True)
class CategoricalZoneDescription:
    """Self-description of a categorical axis: no override param, just classes."""

    kind: ClassVar[Literal['categorical']] = 'categorical'
    classes: tuple[ZoneClassDescription, ...]


# What a scheme's describe() returns -- exactly one member per scheme kind, so
# consumers match on the type instead of reading null-able fields off a bag.
ZoneDescription = (
    BandedZoneDescription
    | BucketedZoneDescription
    | ThresholdZoneDescription
    | CategoricalZoneDescription
)


@dataclass(frozen=True)
class Zone:
    """A single zone along one axis -- one cell of a :class:`ZoneScheme`.

    The base carries the identity every zone shares: ``key`` is a stable id for
    the zone within its axis (a band's ``'<min>_<max>'`` or a class's name) and
    ``label`` is its human label.
    """

    key: str
    label: str

    def ref(self: Self, layer: str) -> ZoneRef:
        """The self-describing :class:`ZoneRef` for this zone on axis ``layer``.

        Each kind builds its own concrete
        :class:`~snowtool.snowdb.zonal_stat_models.ZoneRef` member directly, so a
        crossed query reports a typed per-axis ref without being switched on
        externally.
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

    ``min``/``max`` are integer bounds for an integer-stepped axis (feet for
    elevation) and fractional for a bucketed one (the dimensionless ``[-1, 1]``
    aspect components); ``unit`` is ``None`` when the measure is dimensionless.
    """

    min: int | float
    max: int | float
    unit: str | None

    def __str__(self: Self) -> str:
        return f'{self.min}_{self.max}'

    def ref(self: Self, layer: str) -> BandZoneRef:
        return BandZoneRef(layer=layer, min=self.min, max=self.max, unit=self.unit)

    def csv_columns(self: Self, layer: str) -> list[tuple[str, str]]:
        suffix = f'_{self.unit}' if self.unit else ''
        return [
            (f'{layer}_min{suffix}', str(self.min)),
            (f'{layer}_max{suffix}', str(self.max)),
        ]


@dataclass(frozen=True)
class ClassZone(Zone):
    """One discrete class, identified by its on-disk pixel ``code``."""

    code: int

    def __str__(self: Self) -> str:
        return self.label

    def ref(self: Self, layer: str) -> ClassZoneRef:
        return ClassZoneRef(layer=layer, code=self.code, label=self.label)

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
    side: Literal['below', 'above']

    def __str__(self: Self) -> str:
        return self.label

    def ref(self: Self, layer: str) -> ThresholdZoneRef:
        return ThresholdZoneRef(
            layer=layer,
            threshold=self.threshold,
            unit=self.unit,
            side=self.side,
            label=self.label,
        )

    def csv_columns(self: Self, layer: str) -> list[tuple[str, str]]:
        return [
            (f'{layer}_side', self.label),
            (f'{layer}_threshold_{self.unit}', f'{self.threshold:g}'),
        ]


class ZoneScheme(ABC):
    """How one zone layer's pixels map to zones.

    A scheme is *resolved* to a configured instance before use, then queried with
    no further parameters:

    * :meth:`configured` folds a dataset's configured zone ``params`` (its
      ``zones`` block -- e.g. ``band_step_ft``, ``threshold_pct``) into a new
      scheme instance (the base scheme, e.g. categorical, takes no params and
      returns itself).
    * :meth:`with_override` then folds in an explicit per-query override value
      (the CLI/API ``LAYER:PARAM=VALUE`` token, already typed by
      :meth:`parse_override`).

    After resolution, :meth:`zones` enumerates the scheme's zones in ordinal order
    and :meth:`assign` maps an array of native pixel values to per-pixel zone
    ordinals (``-1`` = out of zone, which uniformly covers layer-nodata and
    out-of-domain values) -- both from the instance's own fields, taking no kwargs.
    """

    @abstractmethod
    def zones(self: Self) -> tuple[Zone, ...]:
        """The scheme's zones, in ordinal order (from the instance's own fields)."""
        raise NotImplementedError

    @abstractmethod
    def assign(
        self: Self,
        values: numpy.typing.NDArray,
    ) -> numpy.typing.NDArray[numpy.int64]:
        """Per-pixel zone ordinal for ``values`` (``-1`` where out of zone)."""
        raise NotImplementedError

    @abstractmethod
    def describe(self: Self) -> ZoneDescription:
        """This scheme's self-description (kind, override param, default, unit).

        The switch-free surface the API discovery/query layers read to advertise a
        zone and accept its override, instead of ``isinstance``-ing the scheme.
        """
        raise NotImplementedError

    def configured(self: Self, params: ZoneLayerParams | None) -> Self:
        """A copy of this scheme with the dataset's configured param applied.

        ``params`` arrives as the specific member model the config parsed to
        (``None`` = unconfigured / no params). The base scheme takes none (a
        categorical axis), so configured params are a config error -- the
        member models make a misplaced param detectable instead of silently
        ignorable.
        """
        if params is not None:
            raise ZoneParamsError(
                f'this zone layer takes no params; got {type(params).__name__}',
            )
        return self

    def with_override(self: Self, override: int | float) -> Self:
        """A copy of this scheme with an explicit per-query ``override`` applied.

        The counterpart of :meth:`configured` for the *explicit* override a
        selection carries (vs. the dataset's configured default). The base scheme
        consumes no override -- :meth:`parse_override` rejects a token for a
        categorical axis, so this is never reached for one.
        """
        raise NotImplementedError

    def parse_override(self: Self, layer_key: str, raw: str) -> int | float:
        """Parse a query's ``:override`` token (the CLI ``--zone`` flag) for this
        scheme. The base scheme takes none, so any token is an error -- a
        categorical axis has nothing to override."""
        raise QueryParameterError(
            f'zone {layer_key!r} takes no override (it is a categorical axis); '
            f'drop the ":{raw}".',
        )


def _as_number(value: float) -> int | float:
    """A band bound as an ``int`` when integral, else a noise-trimmed ``float``.

    Keeps integer-domain bounds (elevation feet) rendering as ``3000`` while letting a
    bucketed axis' fractional bounds render as ``0.5`` -- and rounds so an odd bucket
    count can't leak ``0.30000000000000004`` into a band key/label.
    """
    rounded = round(value, 6)
    return int(rounded) if rounded == int(rounded) else rounded


def _assign_bands(
    values: numpy.typing.NDArray,
    bands: tuple[BandZone, ...],
    value_scale: float,
    layer_nodata: float,
    *,
    closed_top: bool = False,
) -> numpy.typing.NDArray[numpy.int64]:
    """Digitize ``values`` (native units) into ``bands``' ordinals.

    Scales to zone units and digitizes into the band edges; pixels below/above the
    domain, and layer-nodata pixels, become ``-1``. Shared by the step-based
    :class:`BandedZoning` and the count-based :class:`EvenBucketZoning`. ``closed_top``
    folds a value exactly at the top edge into the last band (the bucket scheme tiles a
    closed domain exactly, so its final bucket is ``[.., max]``, not half-open).
    """
    edges = numpy.array(
        [band.min for band in bands] + [bands[-1].max],
        dtype=numpy.float64,
    )
    scaled = numpy.asarray(values, dtype=numpy.float64) * value_scale
    ordinals = (numpy.digitize(scaled, edges) - 1).astype(numpy.int64)
    if closed_top:
        ordinals[scaled == edges[-1]] = len(bands) - 1
    ordinals[(ordinals < 0) | (ordinals >= len(bands))] = -1
    # Belt-and-suspenders: an explicit nodata sentinel is excluded even if it somehow
    # scaled into the domain.
    ordinals[numpy.asarray(values) == layer_nodata] = -1
    return ordinals


@dataclass(frozen=True)
class BandedZoning(ZoneScheme):
    """Contiguous numeric bands aligned to 0 over ``[domain_min, domain_max]``.

    The domain is expressed in *zone* units (e.g. feet for elevation, percent for
    forest cover); ``value_scale`` maps native pixel units to those zone units
    (elevation pixels are metres, so ``value_scale`` is ``M_TO_FT``; forest pixels
    are already percent, so it is ``1``). Bands are aligned to 0 so a given band
    means the same thing regardless of the domain, and ``default_step`` is the band
    width (folded in from the per-dataset ``band_step_ft`` by :meth:`configured` or
    a query ``:override`` by :meth:`with_override`).
    """

    domain_min: float
    domain_max: float
    default_step: int
    unit: str
    value_scale: float
    layer_nodata: float
    # The dataset/query param that configures this scheme's band width.
    param_key: ClassVar[str] = 'band_step_ft'

    def __post_init__(self: Self) -> None:
        if not isinstance(self.default_step, int) or self.default_step <= 0:
            raise ValueError(
                f'band step must be a positive int, got {self.default_step!r}',
            )

    def configured(self: Self, params: ZoneLayerParams | None) -> Self:
        if params is None:
            return self
        if not isinstance(params, BandStepParams):
            raise ZoneParamsError(
                f'a banded zone layer is configured by {self.param_key!r}; '
                f'got {type(params).__name__}',
            )
        return replace(self, default_step=params.band_step_ft)

    def with_override(self: Self, override: int | float) -> Self:
        return replace(self, default_step=override)  # type: ignore[arg-type]

    def parse_override(self: Self, layer_key: str, raw: str) -> int:
        try:
            return int(raw)
        except ValueError as e:
            raise QueryParameterError(
                f'zone {layer_key!r} band step must be an integer, got {raw!r}.',
            ) from e

    def describe(self: Self) -> BandedZoneDescription:
        bands = self.zones()
        return BandedZoneDescription(
            param_key=self.param_key,
            default=self.default_step,
            unit=self.unit,
            min=bands[0].min,
            max=bands[-1].max,
        )

    def zones(self: Self) -> tuple[BandZone, ...]:
        """The bands spanning the domain at ``default_step``.

        Aligned to 0: band ``i`` is ``[i*step, (i+1)*step)``. Reproduces the old
        ``ElevationBand.generate`` for the elevation domain.
        """
        step = self.default_step
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
    ) -> numpy.typing.NDArray[numpy.int64]:
        """Digitize ``values`` (native units) into the band ordinals.

        Scales to zone units and digitizes into the band edges; pixels below/above
        the domain, and layer-nodata pixels, become ``-1``.
        """
        return _assign_bands(values, self.zones(), self.value_scale, self.layer_nodata)


@dataclass(frozen=True)
class EvenBucketZoning(ZoneScheme):
    """A fixed count of equal-width buckets over a closed ``[domain_min, domain_max]``.

    For a bounded, *dimensionless* measure -- the ``[-1, 1]`` aspect components (mean
    ``cos``/``sin`` of aspect) -- where a band *width* carries no external meaning: the
    only useful knob is how many even buckets to cut the range into. So the param is an
    integer ``default_buckets`` (not a width), the bucket bounds are computed (and
    fractional), and there is no unit. Contrast :class:`BandedZoning`, whose fixed step
    aligned to 0 is what an open-ended, real-unit axis (elevation feet) wants.
    """

    domain_min: float
    domain_max: float
    default_buckets: int
    layer_nodata: float
    # The dataset/query param that configures this layer's bucket count.
    param_key: ClassVar[str] = 'buckets'

    def __post_init__(self: Self) -> None:
        if not isinstance(self.default_buckets, int) or self.default_buckets < 1:
            raise ValueError(
                f'bucket count must be a positive int, got {self.default_buckets!r}',
            )

    def configured(self: Self, params: ZoneLayerParams | None) -> Self:
        if params is None:
            return self
        if not isinstance(params, BucketParams):
            raise ZoneParamsError(
                f'a bucketed zone layer is configured by {self.param_key!r}; '
                f'got {type(params).__name__}',
            )
        return replace(self, default_buckets=params.buckets)

    def with_override(self: Self, override: int | float) -> Self:
        return replace(self, default_buckets=int(override))

    def parse_override(self: Self, layer_key: str, raw: str) -> int:
        try:
            return int(raw)
        except ValueError as e:
            raise QueryParameterError(
                f'zone {layer_key!r} bucket count must be an integer, got {raw!r}.',
            ) from e

    def describe(self: Self) -> BucketedZoneDescription:
        bands = self.zones()
        return BucketedZoneDescription(
            param_key=self.param_key,
            default=self.default_buckets,
            min=bands[0].min,
            max=bands[-1].max,
        )

    def zones(self: Self) -> tuple[BandZone, ...]:
        """The domain cut into ``default_buckets`` equal, contiguous buckets."""
        width = (self.domain_max - self.domain_min) / self.default_buckets
        bands = []
        for i in range(self.default_buckets):
            low = _as_number(self.domain_min + i * width)
            high = _as_number(self.domain_min + (i + 1) * width)
            bands.append(
                BandZone(
                    key=f'{low}_{high}',
                    label=f'{low} to {high}',
                    min=low,
                    max=high,
                    unit=None,
                ),
            )
        return tuple(bands)

    def assign(
        self: Self,
        values: numpy.typing.NDArray,
    ) -> numpy.typing.NDArray[numpy.int64]:
        """Digitize ``values`` into the bucket ordinals (dimensionless: no scaling)."""
        return _assign_bands(
            values,
            self.zones(),
            1,
            self.layer_nodata,
            closed_top=True,
        )


@dataclass(frozen=True)
class ThresholdZoning(ZoneScheme):
    """A binary split at a threshold: *below* vs *at-or-above* it.

    The query unit (e.g. forest cover: "below 50% is unforested, 50%+ is
    forested"). ``value_scale`` maps native pixel units to the threshold's unit;
    ``default_threshold`` is the split point (folded in from the per-dataset
    ``threshold_pct`` by :meth:`configured` or a query ``:override`` by
    :meth:`with_override`). The two zones are :class:`ThresholdZone`\\ s (ordinal 0
    below, 1 at-or-above) whose ``threshold`` rides on each cell as a structured
    value so each stays self-describing.
    """

    default_threshold: float
    # The measured quantity's range the split sits within (forest cover 0..100 %,
    # normalised entropy 0..1); advertised as the axis' min/max, not enforced by
    # :meth:`assign` (every value is simply below or at-or-above the threshold).
    domain_min: float
    domain_max: float
    unit: str
    value_scale: float
    layer_nodata: float
    below_label: str
    above_label: str
    # Which threshold param configures this split: forest cover uses
    # ``threshold_pct``; normalised aspect entropy uses ``entropy_threshold``.
    param_key: Literal['threshold_pct', 'entropy_threshold'] = 'threshold_pct'

    def configured(self: Self, params: ZoneLayerParams | None) -> Self:
        if params is None:
            return self
        if self.param_key == 'threshold_pct' and isinstance(params, ThresholdParams):
            return replace(self, default_threshold=params.threshold_pct)
        if self.param_key == 'entropy_threshold' and isinstance(
            params,
            EntropyThresholdParams,
        ):
            return replace(self, default_threshold=params.entropy_threshold)
        raise ZoneParamsError(
            f'this threshold zone layer is configured by {self.param_key!r}; '
            f'got {type(params).__name__}',
        )

    def with_override(self: Self, override: int | float) -> Self:
        return replace(self, default_threshold=float(override))

    def parse_override(self: Self, layer_key: str, raw: str) -> float:
        try:
            return float(raw)
        except ValueError as e:
            raise QueryParameterError(
                f'zone {layer_key!r} threshold must be a number, got {raw!r}.',
            ) from e

    def describe(self: Self) -> ThresholdZoneDescription:
        return ThresholdZoneDescription(
            param_key=self.param_key,
            default=float(self.default_threshold),
            unit=self.unit,
            min=_as_number(self.domain_min),
            max=_as_number(self.domain_max),
        )

    def zones(self: Self) -> tuple[ThresholdZone, ...]:
        """The two sides of the split (below, at-or-above), with clean labels.

        The active threshold rides on each :class:`ThresholdZone` as a structured
        value (not embedded in the label).
        """
        threshold = float(self.default_threshold)
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
    ) -> numpy.typing.NDArray[numpy.int64]:
        """1 where ``values`` (scaled) >= threshold, 0 below, ``-1`` for nodata."""
        threshold = float(self.default_threshold)
        scaled = numpy.asarray(values, dtype=numpy.float64) * self.value_scale
        out = (scaled >= threshold).astype(numpy.int64)
        out[numpy.asarray(values) == self.layer_nodata] = -1
        return out


@dataclass(frozen=True)
class CategoricalZoning(ZoneScheme):
    """A fixed set of discrete classes keyed by their on-disk pixel codes."""

    classes: tuple[ClassZone, ...]
    layer_nodata: int

    def describe(self: Self) -> CategoricalZoneDescription:
        return CategoricalZoneDescription(
            classes=tuple(
                ZoneClassDescription(key=cls.key, label=cls.label)
                for cls in self.classes
            ),
        )

    def zones(self: Self) -> tuple[ClassZone, ...]:
        """The class list (its order *is* the ordinal order)."""
        return self.classes

    def assign(
        self: Self,
        values: numpy.typing.NDArray,
    ) -> numpy.typing.NDArray[numpy.int64]:
        """Map each pixel's class code to its ordinal (``-1`` for nodata/unknown).

        Layer-nodata and any code not in :attr:`classes` fall through to ``-1``.
        """
        out = numpy.full(numpy.shape(values), -1, dtype=numpy.int64)
        for ordinal, cls in enumerate(self.classes):
            out[numpy.asarray(values) == cls.code] = ordinal
        return out
