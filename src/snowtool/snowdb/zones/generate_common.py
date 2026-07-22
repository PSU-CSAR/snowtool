"""Shared plumbing for the zone-layer generation engines (terrain, land cover, ...).

:mod:`~snowtool.snowdb.zones.terrain_generate` and
:mod:`~snowtool.snowdb.zones.landcover_generate` each stream one fine-resolution
source once and bin it into every target grid, but disagree on everything about
*what* gets derived per pixel (slope/aspect vs. a forest/valid count). What they
share is the machinery around that: a pre-flight existence guard, the
generation-digest-then-stamp pass that turns per-target finalized arrays into one
provenance hash, and the point-in-cell binning arithmetic (cell assignment,
pixel-centre coordinates). Extracted here so a third provider does not have to
copy any of it a third time.
"""

from __future__ import annotations

import hashlib
import math

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy
import rasterio

from snowtool.exceptions import ArtifactExistsError
from snowtool.snowdb.provenance import versioned_hash
from snowtool.snowdb.raster.cog import write_cog

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy.typing

    from affine import Affine

    from snowtool.snowdb.zones.zone_layer import ZoneLayer, ZoneLayerTarget


class GenerationAccumulator(Protocol):
    """What :func:`finalize_and_stamp` needs from one target's accumulator.

    Both :class:`~snowtool.snowdb.zones.terrain_generate._GridAccumulator` and
    :class:`~snowtool.snowdb.zones.landcover_generate._ForestAccumulator`
    satisfy this structurally (no inheritance needed): a ``target`` to name and
    sort by, and a ``finalize`` that reduces the accumulator to its per-target
    layer/array pairs. The generation hash is defined over the *first* pair's
    array (terrain: the elevation array, first in its list; land cover: the
    forest-pct array, its only pair), so each engine controls the digested
    array purely by ordering its ``finalize`` output.
    """

    target: ZoneLayerTarget

    def finalize(self) -> list[tuple[ZoneLayer, numpy.typing.NDArray]]: ...


@dataclass(frozen=True)
class Block:
    """A nominal streaming block, in row-major enumeration order.

    ``(c0, r0)`` is the block's col/row origin within the grid (or window) being
    streamed; ``bw``/``bh`` are its width/height, clamped at the trailing edge.
    Shared by both generation engines so their block enumeration is identical.
    """

    c0: int
    r0: int
    bw: int
    bh: int


def iter_blocks(
    width: int,
    height: int,
    block_size: int,
    *,
    col_off: int = 0,
    row_off: int = 0,
) -> list[Block]:
    """Enumerate ``width`` x ``height`` (from ``col_off``/``row_off``) in blocks.

    Row-major order (``for by ... for bx ...``), exactly matching both engines'
    prior inline enumeration, so switching either engine onto this helper leaves
    its output -- and generation hash -- bit-identical. ``col_off``/``row_off``
    cover landcover's source-window offset; terrain streams from the origin and
    passes the defaults.
    """
    nbx = math.ceil(width / block_size)
    nby = math.ceil(height / block_size)
    blocks: list[Block] = []
    for by in range(nby):
        for bx in range(nbx):
            c0 = col_off + bx * block_size
            r0 = row_off + by * block_size
            blocks.append(
                Block(
                    c0=c0,
                    r0=r0,
                    bw=min(block_size, col_off + width - c0),
                    bh=min(block_size, row_off + height - r0),
                ),
            )
    return blocks


def require_absent_layers(
    targets: Iterable[ZoneLayerTarget],
    layers: Iterable[ZoneLayer],
    kind: str,
) -> None:
    """Refuse to generate ``kind`` layers over any target that already has them.

    Checked across every target before the caller's (potentially large/expensive)
    source read -- callers only reach this when not ``force``. ``kind`` (e.g.
    ``'terrain'`` / ``'land cover'``) changes only the message wording; the two
    engines' error text is otherwise identical.
    """
    layers = list(layers)
    for target in targets:
        existing = [
            layer.filename
            for layer in layers
            if (target.directory / layer.filename).is_file()
        ]
        if existing:
            raise ArtifactExistsError(
                f'Could not generate {kind} for {target.name}: '
                f'{target.directory} already has {", ".join(existing)}. '
                'Remove and try again or use force=True.',
            )


def finalize_and_stamp(
    accumulators: Iterable[GenerationAccumulator],
    *,
    format_version: int,
    hash_tag: str,
) -> dict[str, str]:
    """Finalize every accumulator, compute one generation hash, then write.

    One generation id for the whole streaming pass: a sha256 digest over every
    target's name plus its *first* finalized layer's array (sorted by target
    name for determinism), turned into a
    :func:`~snowtool.snowdb.provenance.versioned_hash` and stamped identically
    (under ``hash_tag``) on every output of every target -- so everything
    produced together reconciles as one set. The iteration order (sorted for the
    digest, input order for the returned mapping), the per-accumulator update
    sequence (name bytes, then array bytes), and which array is digested are all
    provenance-visible and must stay exactly as each caller already relies on:
    the digested array is definitionally the first pair each ``finalize``
    returns (terrain lists elevation first; land cover returns only the forest
    array), so each engine pins its digested array by ordering that list.
    """
    accs = list(accumulators)
    finalized: list[
        tuple[GenerationAccumulator, list[tuple[ZoneLayer, numpy.typing.NDArray]]]
    ] = []
    digest = hashlib.sha256()
    for acc in sorted(accs, key=lambda acc: acc.target.name):
        artifacts = acc.finalize()
        finalized.append((acc, artifacts))
        digest.update(acc.target.name.encode('utf-8'))
        digest.update(artifacts[0][1].tobytes())
    generation_hash = versioned_hash(format_version, digest.hexdigest())

    for acc, artifacts in finalized:
        write_layers(acc.target, artifacts, generation_hash, hash_tag)
    return dict.fromkeys((acc.target.name for acc in accs), generation_hash)


def write_layers(
    target: ZoneLayerTarget,
    artifacts: list[tuple[ZoneLayer, numpy.typing.NDArray]],
    tag: str,
    hash_tag: str,
) -> None:
    """Write each finalized layer as its own COG, ``tag`` stamped at ``hash_tag``."""
    target.directory.mkdir(parents=True, exist_ok=True)
    rio_crs = rasterio.crs.CRS.from_wkt(target.crs.to_wkt())
    for layer, array in artifacts:
        write_cog(
            target.directory / layer.filename,
            array,
            transform=target.transform,
            crs=rio_crs,
            nodata=layer.nodata,
            tile_size=target.tile_size,
            band_descriptions=layer.band_descriptions,
            tags={hash_tag: tag},
        )


def cells_for_points(
    inv_transform: Affine,
    xt: numpy.typing.NDArray[numpy.float64],
    yt: numpy.typing.NDArray[numpy.float64],
    width: int,
    height: int,
) -> tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.bool_]]:
    """Assign points already in a target grid's CRS to that grid's cells.

    ``inv_transform`` is the target's inverse affine (``~transform``). Returns
    ``(cell, in_bounds)``: ``cell`` is the flattened ``row * width + col`` index
    for *every* input point, computed before masking, so out-of-bounds entries
    are meaningless placeholders the caller discards via ``in_bounds`` (e.g.
    ``cell[in_bounds]``) -- purely elementwise arithmetic, so this is identical
    to filtering ``row``/``col`` first. The expression order (``a * xt + b * yt +
    c``, floor, then cast to int64) matches both call sites' prior inline code
    exactly, keeping every existing generation hash bit-identical.
    """
    col = numpy.floor(
        inv_transform.a * xt + inv_transform.b * yt + inv_transform.c,
    ).astype(numpy.int64)
    row = numpy.floor(
        inv_transform.d * xt + inv_transform.e * yt + inv_transform.f,
    ).astype(numpy.int64)
    in_bounds = (col >= 0) & (col < width) & (row >= 0) & (row < height)
    cell = row * width + col
    return cell, in_bounds


def pixel_centre_coords(
    transform: Affine,
    r0: int,
    c0: int,
    height: int,
    width: int,
) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]:
    """Pixel-centre ``(x, y)`` coordinates for a ``height`` x ``width`` block.

    ``(r0, c0)`` is the block's row/col origin within the full grid ``transform``
    covers. Returns broadcastable ``(height, 1)`` / ``(1, width)`` arrays rather
    than a dense meshgrid, matching both call sites' prior inline code exactly
    (same affine expansion, same broadcasting shape).
    """
    rows = (numpy.arange(height) + r0)[:, None]
    cols = (numpy.arange(width) + c0)[None, :]
    x = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    y = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    return x, y
