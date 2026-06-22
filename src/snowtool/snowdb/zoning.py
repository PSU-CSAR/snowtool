"""The zone model: how a layer's pixels map to zones.

A :class:`ZoneScheme` declares the zones a
:class:`~snowtool.snowdb.zone_layer.ZoneLayer` stratifies the grid into and how
each pixel is assigned to one. This module holds only the abstract contract and
the per-axis :class:`Zone` descriptors; the concrete schemes (banded +
categorical) and the query-time registry are layered on in the zone-model phase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing


@dataclass(frozen=True)
class Zone:
    """A single zone along one axis -- one cell of a :class:`ZoneScheme`.

    The base carries the identity every zone shares (its ``key`` and human
    ``label``); banded/categorical refinements add their own bounds/code.
    """

    key: str
    label: str


class ZoneScheme(ABC):
    """How one zone layer's pixels map to zones.

    :meth:`zones` enumerates the scheme's zones (optionally overriding its
    defaults), and :meth:`assign` maps an array of native pixel values to per-pixel
    zone ordinals (``-1`` = out of zone, which uniformly covers layer-nodata and
    out-of-domain values).
    """

    @abstractmethod
    def zones(self, **override: object) -> tuple[Zone, ...]:
        """The scheme's zones, in ordinal order."""
        raise NotImplementedError

    @abstractmethod
    def assign(
        self,
        values: numpy.typing.NDArray,
        **override: object,
    ) -> numpy.typing.NDArray:
        """Per-pixel zone ordinal for ``values`` (``-1`` where out of zone)."""
        raise NotImplementedError
