"""Shared plumbing for the zone-layer generation engines (terrain, land cover, ...).

:mod:`~snowtool.snowdb.zones.terrain_generate` and
:mod:`~snowtool.snowdb.zones.landcover_generate` each stream one fine-resolution
source once and bin it into every target grid, but disagree on everything about
*what* gets derived per pixel (slope/aspect vs. a forest/valid count). What they
share is the machinery around that: a pre-flight existence guard, the
generation-digest-then-stamp pass that turns per-target finalized arrays into one
provenance hash, the point-in-cell binning arithmetic (cell assignment,
pixel-centre coordinates), the per-target accumulator prologue
(:class:`BinAccumulator`), and the whole streaming concurrency scaffold
(:class:`StreamingBinner`: the read lock, the thread-local per-target
Transformers, the cancel/re-check-under-lock dance, the coordinate
broadcast/mask/per-target-transform epilogue, the serial ordered reduce, and the
``ordered_parallel_map`` wiring).
"""

from __future__ import annotations

import hashlib
import math
import threading

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy
import rasterio

from pyproj import Transformer

from snowtool.exceptions import ArtifactExistsError
from snowtool.snowdb.progress import NULL_PROGRESS, ProgressReporter
from snowtool.snowdb.provenance import versioned_hash
from snowtool.snowdb.raster.cog import write_cog
from snowtool.snowdb.zones.parallel import (
    CancelToken,
    ordered_parallel_map,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    import numpy.typing

    from affine import Affine

    from snowtool.snowdb.zones.zone_layer import ZoneLayer, ZoneLayerTarget


class BinAccumulator(ABC):
    """Per-target point-in-cell accumulator: the shared bin-geometry prologue.

    Both engines' accumulators pin one target grid and bin fine source pixels
    into its cells; all that differs is *what* is summed per cell (terrain: class
    counts + cos/sin/z sums; land cover: forest/valid counts). This base owns the
    identical prologue -- the target, its ``rows``/``cols``/``transform``/``crs``,
    the inverse affine ``_inv`` its ``bin_into`` needs, and the ``_ncell`` flat
    size -- so a subclass only allocates its own count arrays and defines
    ``bin_into``/``finalize``.
    """

    def __init__(self: Self, target: ZoneLayerTarget) -> None:
        self.target = target
        self.height = target.rows
        self.width = target.cols
        self.transform = target.transform
        # The streamer builds (thread-local) source/work-CRS -> this-CRS
        # Transformers, so ``bin_into`` receives coords already in this grid's CRS.
        self.crs = target.crs
        self._inv = ~self.transform

    @property
    def _ncell(self: Self) -> int:
        return self.height * self.width

    @abstractmethod
    def bin_into(
        self: Self,
        xt: numpy.typing.NDArray[numpy.float64],
        yt: numpy.typing.NDArray[numpy.float64],
        *payload: numpy.typing.NDArray,
    ) -> None:
        """Bin one block's already-reprojected (this grid's CRS) pixels into cells.

        ``xt``/``yt`` are the kept pixel centres in this grid's CRS; ``payload`` is
        the engine's per-pixel arrays (the tuple the streamer splats in), aligned
        with them. Runs serially on the main thread in fixed block order, so the
        accumulation order is identical to the serial pass -- keeping the
        generation hash reproducible regardless of worker count.
        """

    @abstractmethod
    def finalize(
        self: Self,
    ) -> list[tuple[ZoneLayer, numpy.typing.NDArray]]: ...


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

    Row-major order (``for by ... for bx ...``) is provenance-visible: it feeds the
    generation hash, so it must stay stable. ``col_off``/``row_off`` cover
    landcover's source-window offset; terrain streams from the origin and passes
    the defaults.
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
    accumulators: Iterable[BinAccumulator],
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

    The write is two-phase so a crash cannot strand a partial layer set that
    :func:`require_absent_layers` would then refuse to regenerate: every COG of
    every target is first written to a ``.part`` sidecar, and only after all of
    them exist are they renamed into place. A failure during the (long) write
    phase leaves only sidecars -- no final layer file -- so the next run
    regenerates cleanly, overwriting the stale sidecars. The commit phase is a
    handful of atomic renames, never a partially-written file.
    """
    accs = list(accumulators)
    finalized: list[
        tuple[BinAccumulator, list[tuple[ZoneLayer, numpy.typing.NDArray]]]
    ] = []
    digest = hashlib.sha256()
    for acc in sorted(accs, key=lambda acc: acc.target.name):
        artifacts = acc.finalize()
        finalized.append((acc, artifacts))
        digest.update(acc.target.name.encode('utf-8'))
        digest.update(artifacts[0][1].tobytes())
    generation_hash = versioned_hash(format_version, digest.hexdigest())

    pending: list[tuple[Path, Path]] = []
    for acc, artifacts in finalized:
        pending.extend(write_layers(acc.target, artifacts, generation_hash, hash_tag))
    for part, final in pending:
        part.replace(final)
    return dict.fromkeys((acc.target.name for acc in accs), generation_hash)


def write_layers(
    target: ZoneLayerTarget,
    artifacts: list[tuple[ZoneLayer, numpy.typing.NDArray]],
    tag: str,
    hash_tag: str,
) -> list[tuple[Path, Path]]:
    """Write each finalized layer as a ``.part`` sidecar COG, ``tag`` at ``hash_tag``.

    Returns the ``(sidecar, final)`` path pairs for the caller
    (:func:`finalize_and_stamp`) to commit once every target's sidecars exist,
    so no final layer file ever appears before the whole generation succeeded.
    """
    target.directory.mkdir(parents=True, exist_ok=True)
    rio_crs = rasterio.crs.CRS.from_wkt(target.crs.to_wkt())
    pending: list[tuple[Path, Path]] = []
    for layer, array in artifacts:
        final = target.directory / layer.filename
        part = final.with_suffix(final.suffix + '.part')
        write_cog(
            part,
            array,
            transform=target.transform,
            crs=rio_crs,
            nodata=layer.nodata,
            tile_size=target.tile_size,
            band_descriptions=layer.band_descriptions,
            tags={hash_tag: tag},
        )
        pending.append((part, final))
    return pending


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
    c``, floor, then cast to int64) is provenance-visible and must not change.
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
    than a dense meshgrid; the affine expansion is provenance-visible and must not
    change.
    """
    rows = (numpy.arange(height) + r0)[:, None]
    cols = (numpy.arange(width) + c0)[None, :]
    x = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    y = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    return x, y


type _F64 = numpy.typing.NDArray[numpy.float64]
# A block's per-pixel payload arrays the reducer splats into ``bin_into`` (terrain:
# cls/cos/sin/z; land cover: is_forest).
type Payload = tuple[numpy.typing.NDArray, ...]
# See :meth:`StreamingBinner._load`.
type Loaded = tuple[Payload, _F64, _F64, numpy.typing.NDArray[numpy.bool_]] | None
# See :meth:`StreamingBinner._compute`.
type BlockResult = tuple[Payload, list[tuple[_F64, _F64]]]


class StreamingBinner[Acc: BinAccumulator](ABC):
    """Streams one source in blocks, binning into every target accumulator.

    The concurrency scaffold shared by every input-driven scatter engine
    (terrain, land cover, ...). The expensive per-block work -- the subclass's
    read/derivation plus the per-target pyproj reprojection -- is pure and runs on
    a worker pool. The only shared mutable state is the accumulators, so binning is
    done serially on the main thread in streaming block order; that keeps
    accumulation order independent of worker count, so the generation hash is
    reproducible (parallel == serial bit for bit). Each worker thread gets its own
    Transformers (a Transformer is not safe to share concurrently).

    Block reads must run under :attr:`_read_lock` (via :meth:`_locked_read`): a
    GDAL dataset (and any shared WarpedVRT over it) is not safe for concurrent
    reads. The read is a small fraction of the per-block cost (the reprojection
    dominates and still runs fully in parallel), so serialising it costs little.

    The parallel-map / serial-reduce machinery -- the sliding window, the
    warm-gate, and the ctrl+c-proof teardown that guarantees no worker is left
    inside the source dataset the caller closes on return -- lives in
    :mod:`snowtool.snowdb.zones.parallel`; :meth:`run` just wires
    :meth:`_compute`/:meth:`_reduce` into it.

    A subclass supplies only what differs: ``_source_crs`` (the CRS the derived
    pixel centres are in, transformed to each target), the block enumeration
    (:meth:`_blocks`), the per-block read+derive (:meth:`_load`), and the progress
    label (:attr:`_label`).
    """

    #: Progress-bar label for this engine's streaming pass.
    _label: str

    def __init__(
        self: Self,
        source_crs: str,
        accumulators: list[Acc],
    ) -> None:
        self._source_crs = source_crs
        self._accumulators = accumulators
        self._local = threading.local()
        # GDAL/WarpedVRT reads are not concurrency-safe; serialise just the read.
        self._read_lock = threading.Lock()

    def _transformers(self: Self) -> list[Transformer]:
        """Per-thread source-CRS -> target-CRS Transformers (built once per thread)."""
        tfs: list[Transformer] | None = getattr(self._local, 'tfs', None)
        if tfs is None:
            tfs = [
                Transformer.from_crs(self._source_crs, acc.crs, always_xy=True)
                for acc in self._accumulators
            ]
            self._local.tfs = tfs
        return tfs

    def _locked_read[R](
        self: Self,
        cancel: CancelToken,
        read: Callable[[], R],
    ) -> R | None:
        """Run ``read`` under the read lock, or ``None`` if the run is aborting.

        Bail before queueing on the lock: once the run is aborting, blocks waiting
        their turn must not each still pay a read. The re-check under the lock
        closes the race for a worker that passed the first check just before
        cancellation and acquired the lock just after it.
        """
        if cancel.cancelled:
            return None
        with self._read_lock:
            if cancel.cancelled:
                return None
            return read()

    @abstractmethod
    def _blocks(self: Self) -> list[Block]:
        """The blocks to stream, in row-major order (see :func:`iter_blocks`)."""

    @abstractmethod
    def _load(self: Self, block: Block, cancel: CancelToken) -> Loaded:
        """Read (via :meth:`_locked_read`) and derive one block. Pure, no shared writes.

        Returns a :data:`Loaded` -- ``(payload, x, y, keep)`` or ``None``. The
        payload arrays are the *unmasked*, block-shaped per-pixel derivations;
        :meth:`_compute` owns the masking, flattening each payload array and the
        broadcast centres by the single ``keep`` mask before transforming the kept
        centres into every target's CRS and splatting the payload into each
        accumulator's ``bin_into``. ``None`` means no contribution (an all-invalid
        block, or a read the cancel short-circuited).
        """

    def _compute(self: Self, block: Block, cancel: CancelToken) -> BlockResult | None:
        """Worker step: read (locked), derive, mask, reproject. Pure, no shared writes.

        Owns the single ``keep`` mask for the whole block: it flattens and applies
        it to every payload array *and* to the pixel centres, so the payload and the
        reprojected coordinates stay aligned by construction (an engine returns the
        unmasked block-shaped payload; only this method masks). Elementwise, so it is
        identical to an engine pre-masking its payload with the same mask.
        """
        loaded = self._load(block, cancel)
        if loaded is None:
            return None
        payload, x, y, keep = loaded
        keep = keep.ravel()
        payload = tuple(arr.ravel()[keep] for arr in payload)
        shape = (x.shape[0], y.shape[1])
        xf = numpy.broadcast_to(x, shape).ravel()[keep]
        yf = numpy.broadcast_to(y, shape).ravel()[keep]
        coords = [tf.transform(xf, yf) for tf in self._transformers()]
        return payload, coords

    def _reduce(self: Self, result: BlockResult) -> None:
        """Main-thread step: bin one block into every accumulator (serial, ordered)."""
        payload, coords = result
        for acc, (xt, yt) in zip(self._accumulators, coords, strict=True):
            acc.bin_into(xt, yt, *payload)

    def run(
        self: Self,
        workers: int,
        progress: ProgressReporter = NULL_PROGRESS,
    ) -> None:
        """Stream all blocks through the ordered parallel-map engine (see class doc)."""
        ordered_parallel_map(
            self._blocks(),
            self._compute,
            self._reduce,
            workers=workers,
            progress=progress,
            label=self._label,
        )
