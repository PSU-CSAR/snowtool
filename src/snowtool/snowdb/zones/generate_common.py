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

from typing import TYPE_CHECKING

import numpy

from snowtool.snowdb.provenance import versioned_hash

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import numpy.typing

    from affine import Affine

    from snowtool.snowdb.zones.zone_layer import ZoneLayer, ZoneLayerTarget


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
            raise FileExistsError(
                f'Could not generate {kind} for {target.name}: '
                f'{target.directory} already has {", ".join(existing)}. '
                'Remove and try again or use force=True.',
            )


def finalize_and_stamp[Acc, Artifacts](
    accumulators: Iterable[Acc],
    *,
    name_of: Callable[[Acc], str],
    finalize: Callable[[Acc], Artifacts],
    digest_array: Callable[[Artifacts], numpy.typing.NDArray],
    write: Callable[[Acc, Artifacts, str], None],
    format_version: int,
) -> dict[str, str]:
    """Finalize every accumulator, compute one generation hash, then write.

    One generation id for the whole streaming pass: a sha256 digest over every
    target's name plus its finalized ``digest_array`` (sorted by target name for
    determinism), turned into a :func:`~snowtool.snowdb.provenance.versioned_hash`
    and stamped identically on every output of every target -- so everything
    produced together reconciles as one set. The iteration order (sorted for the
    digest, input order for the returned mapping), the per-accumulator update
    sequence (name bytes, then array bytes), and which array is digested are all
    provenance-visible and must stay exactly as each caller already relies on
    (terrain digests only the elevation array; land cover digests the forest
    array) -- that is why ``digest_array`` is a caller-supplied projection rather
    than "the finalized artifacts" themselves.
    """
    accs = list(accumulators)
    finalized: list[tuple[Acc, Artifacts]] = []
    digest = hashlib.sha256()
    for acc in sorted(accs, key=name_of):
        artifacts = finalize(acc)
        finalized.append((acc, artifacts))
        digest.update(name_of(acc).encode('utf-8'))
        digest.update(digest_array(artifacts).tobytes())
    generation_hash = versioned_hash(format_version, digest.hexdigest())

    for acc, artifacts in finalized:
        write(acc, artifacts, generation_hash)
    return dict.fromkeys((name_of(acc) for acc in accs), generation_hash)


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
