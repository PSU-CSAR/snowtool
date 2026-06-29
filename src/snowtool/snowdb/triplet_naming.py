"""The station-triplet <-> filename-stem codec used across snowdb storage.

A pourpoint's station triplet is its identity, but ``:`` is not path-safe, so the
on-disk artifacts keyed by a pourpoint encode the triplet with ``_`` instead. This
is the single encoding rule shared by the pourpoint record files
(``pourpoints/records/<stem>.geojson``) and the per-dataset burned AOI rasters
(``<stem>.tif``); both must agree on it for the ``pourpoint sync`` prune diff and
the raster cascade to line up. It is storage naming, not a type -- hence its own
module rather than a home in :mod:`snowtool.types`.
"""

from __future__ import annotations

from snowtool.types import StationTriplet


def triplet_to_stem(triplet: StationTriplet) -> str:
    """The filename stem for a station triplet (``:`` is not path-safe -> ``_``).

    Inverse of :func:`stem_to_triplet`.
    """
    return triplet.replace(':', '_')


def stem_to_triplet(stem: str) -> StationTriplet:
    """The station triplet encoded in a record/raster filename stem (``_`` -> ``:``).

    Inverse of :func:`triplet_to_stem`. Lossless because a valid triplet never
    contains ``_`` (see :data:`~snowtool.types.STATION_TRIPLET`); the result is a
    plain ``str`` (the runtime form of ``StationTriplet``), not a re-validated
    value.
    """
    return stem.replace('_', ':')
