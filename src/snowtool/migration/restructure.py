"""Move a legacy flat ``rasterdb`` directory into the multi-dataset snowdb
layout.

The old layout was a single flat directory (``aoi-rasters/``, ``areas.tif``,
``dem.tif``, ``cogs/``). The snowdb root is ``aois/`` + ``data/<name>/`` per
dataset. This relocates a flat directory to ``data/<dataset>/`` and creates the
global ``aois/`` directory. Driven by ``snowtool migration restructure``.
"""

from __future__ import annotations

import shutil

from pathlib import Path


def restructure_to_snowdb(src: Path, dst_root: Path, dataset: str) -> Path:
    """Move flat dataset dir ``src`` to ``dst_root/data/<dataset>/``.

    Also creates the global ``dst_root/aois/`` directory. Returns the new
    dataset directory. Raises if ``src`` is not a directory or the destination
    dataset directory already exists.
    """
    src = Path(src)
    dst_root = Path(dst_root)

    if not src.is_dir():
        raise ValueError(f'Source is not a directory: {src}')

    dataset_dir = dst_root / 'data' / dataset
    if dataset_dir.exists():
        raise FileExistsError(f'Destination already exists: {dataset_dir}')

    (dst_root / 'data').mkdir(parents=True, exist_ok=True)
    (dst_root / 'aois').mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dataset_dir))

    return dataset_dir
