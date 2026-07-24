"""The *terrain* zone-layer provider: elevation + aspect layers from a DEM.

Because aspect cannot be resampled after the fact (it must be computed from
elevation at the source resolution), terrain is generated once from a fine-resolution
DEM (see :mod:`snowtool.snowdb.zones.terrain_generate`) into a small family of
co-registered layers on the dataset grid, stored under ``data/<name>/terrain/``:

* ``elevation.tif`` -- ``float32`` mean elevation (m).
* ``aspect_majority.tif`` -- ``uint8`` majority aspect class per cell
  (``0`` N, ``1`` E, ``2`` S, ``3`` W, ``4`` flat); nodata ``255``.
* ``northness.tif`` / ``eastness.tif`` -- two ``float32`` single-band layers,
  ``northness`` = mean ``cos(aspect)`` and ``eastness`` = mean ``sin(aspect)``
  over the cell's non-flat pixels (the first circular moment; ``hypot(northness,
  eastness)`` is the orientation purity in ``[0, 1]``). Each is its own query-able
  zone axis, banded over ``[-1, 1]`` (see :data:`NORTHNESS`/:data:`EASTNESS`); a
  cell with no non-flat pixels carries the finite :data:`ASPECT_COMPONENT_NODATA`
  sentinel. Two single-band files rather than one two-band file because the
  :class:`~snowtool.snowdb.zones.zone_layer.ZoneLayer` model is one file + one
  band + one zoning scheme per query key, and the tiled reader reads band 0 only.
* ``aspect_entropy.tif`` -- ``float32`` normalised Shannon entropy of the cell's
  aspect-class distribution (the same five N/E/S/W/flat counts that feed the
  majority vote), in ``[0, 1]``: ``0`` = every pixel one class (coherent),
  ``1`` = evenly mixed; nodata ``-1``. Thresholded into a high-/low-signal zone
  so a query can keep only cells whose majority aspect is well-supported.

Every layer carries a :data:`~snowtool.snowdb.constants.DEM_HASH_TAG` tag -- the
sha256 of the generated elevation array -- so the whole set's provenance can be
read back cheaply.

:func:`terrain_provider` builds the
:class:`~snowtool.snowdb.zones.zone_layer.ZoneLayerProvider` record for this kind
(the layer/format-version definitions live in ``terrain_layers`` so the engine can
import them without importing this module): it only wires them up plus the DEM
source constructors and the engine, so a dataset builds and reads terrain like any
other zone layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from snowtool.snowdb.constants import DEM_HASH_TAG
from snowtool.snowdb.zones.terrain_generate import generate_terrain
from snowtool.snowdb.zones.terrain_layers import (
    TERRAIN_FORMAT_VERSION,
    TERRAIN_LAYERS,
)
from snowtool.snowdb.zones.zone_layer import ZoneLayerProvider

if TYPE_CHECKING:
    from pathlib import Path

    from snowtool.snowdb.zones.zone_layer import GenerationEngine, ZoneLayerSource


def _default_source(root: Path) -> ZoneLayerSource:
    """The default DEM source -- USGS 3DEP streamed from the public bucket."""
    from snowtool.snowdb.zones.terrain_source import ThreeDEP

    return ThreeDEP()


def _local_source(path: Path) -> ZoneLayerSource:
    """A local on-disk DEM file source (the ``--source terrain PATH`` path)."""
    from snowtool.snowdb.zones.terrain_source import LocalFile

    return LocalFile(path)


def terrain_provider(engine: GenerationEngine = generate_terrain) -> ZoneLayerProvider:
    """The terrain zone-layer kind: elevation + aspect, derived from a DEM.

    ``engine`` is the test seam: the default is the real streaming engine, and a
    test passes a fast stand-in that is signature-checked against it.
    """
    return ZoneLayerProvider(
        name='terrain',
        subdir='terrain',
        layers=TERRAIN_LAYERS,
        hash_tag=DEM_HASH_TAG,
        format_version=TERRAIN_FORMAT_VERSION,
        engine=engine,
        default_source=_default_source,
        local_source=_local_source,
    )
