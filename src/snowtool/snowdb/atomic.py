"""Atomic write helpers: crash-consistent persistence for snowdb's on-disk files.

Every persisted file in a snowdb -- pourpoint records, the pourpoint index, the
root/dataset configs, per-date COG directories -- is read back by a later
process (a query, a reindex, the API at startup). A write that dies partway
through (a crash, ENOSPC, ^C) must never leave a *torn* file or directory in
that file's place: readers should see either the old content or the new
content, never a truncated mix of both. These helpers give every writer that
guarantee via the same primitive: write into a temp path that lives beside the
destination (same directory, so guaranteed same filesystem), then
``os.replace`` it onto the destination. A same-filesystem rename is atomic on
POSIX (and on Windows, via ``os.replace`` rather than ``os.rename``), so the
swap itself can never be observed half-done.

This is *not* durability against power loss: there is no ``fsync`` here, so a
kernel/power failure immediately after a successful rename can still lose the
write to volatile page cache. The goal is crash-consistency of *content* --
what ends up on disk is always a complete prior version or a complete new
version, never a partial one -- not guaranteeing the new version survives a
hard power cut. snowdb is not a durability-critical store, so paying for
fsync ceremony on every write is not worth it.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically.

    Writes to a uniquely named temp file in ``path.parent`` (guaranteeing the
    same filesystem as ``path``, so the final swap is an atomic rename), then
    ``os.replace``s it onto ``path``. On any failure -- including one raised by
    ``os.replace`` itself -- the temp file is removed and ``path`` is left
    exactly as it was.
    """
    path = Path(path)
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f'.{path.name}.',
        suffix='.tmp',
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            f.write(text)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_copy(src: Path, dest: Path) -> None:
    """Copy ``src`` to ``dest`` atomically, byte-for-byte.

    Same pattern as :func:`atomic_write_text`, via ``shutil.copyfile`` instead
    of a text write: copy into a uniquely named temp file beside ``dest``, then
    ``os.replace`` it into place. Used where a source file (e.g. a pourpoint
    record's source geojson) is imported verbatim rather than re-serialized, so
    it must not be re-encoded in the process.
    """
    src = Path(src)
    dest = Path(dest)
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=dest.parent,
        prefix=f'.{dest.name}.',
        suffix='.tmp',
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copyfile(src, tmp_path)
        tmp_path.replace(dest)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


@contextmanager
def staged_dir(dest: Path) -> Iterator[Path]:
    """Stage a whole directory tree, then swap it onto ``dest``.

    Yields a fresh, empty temp directory created beside ``dest`` (same parent,
    so every rename below stays on one filesystem). The caller populates the
    yielded directory however it likes. On clean exit:

    - if ``dest`` already exists, it is renamed to a uniquely named
      ``dest.old-*`` sibling (an atomic rename -- same filesystem, directory to
      directory);
    - the temp directory is renamed onto ``dest`` (also atomic);
    - the ``dest.old-*`` directory, now orphaned, is removed with
      ``shutil.rmtree``.

    On an exception raised from within the ``with`` block, the temp directory
    is removed with ``shutil.rmtree`` and ``dest`` is left completely
    untouched -- the caller's partial work never becomes visible.

    POSIX has no primitive that atomically swaps two non-empty directories in
    one step, so there is an unavoidable, brief window between the two renames
    above during which nothing exists at ``dest``. This is a deliberate,
    documented tradeoff rather than a bug: the window sits *between* two
    individually atomic renames, so a concurrent reader can never observe a
    *partially written* directory -- only the wholly-old tree, a (typically
    sub-millisecond) gap, or the wholly-new tree. That is sufficient for
    crash-consistency of content even though the whole swap is not itself a
    single atomic operation.
    """
    dest = Path(dest)
    tmp_dir = Path(
        tempfile.mkdtemp(dir=dest.parent, prefix=f'.{dest.name}.', suffix='.tmp'),
    )
    try:
        yield tmp_dir
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    else:
        had_old = dest.exists()
        old_dir = dest.parent / f'.{dest.name}.old-{uuid.uuid4().hex}'
        if had_old:
            dest.replace(old_dir)
        tmp_dir.replace(dest)
        if had_old:
            shutil.rmtree(old_dir)
