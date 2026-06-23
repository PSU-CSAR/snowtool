"""Stamp a root config onto a legacy snowdb root.

A snowdb built before the root config existed is a bare ``aois/`` + ``data/``
tree with no ``snowdb_conf.json``. :meth:`~snowtool.snowdb.db.SnowDb.open` now
*requires* that config, so such a root must be migrated by writing one. This
stamps a default config (current creation time, the conventional link locations,
no datasets registered) into an existing root. Driven by ``snowtool migration
stamp``; idempotent (an existing config is left untouched).
"""

from __future__ import annotations

from pathlib import Path

from snowtool.snowdb.config import CONFIG_FILENAME, RootConfig


def stamp_root(root: Path) -> tuple[Path, bool]:
    """Write a default ``snowdb_conf.json`` into existing snowdb ``root``.

    Returns ``(config_path, written)`` -- ``written`` is ``False`` when a config
    was already present (left as is) and ``True`` when one was created. Raises if
    ``root`` is not a directory or lacks the base ``aois/``/``data/`` layout (it
    would not be a snowdb to stamp).
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f'Not a directory: {root}')
    missing = [name for name in ('aois', 'data') if not (root / name).is_dir()]
    if missing:
        raise ValueError(
            f'{root} is not a snowdb root (missing {", ".join(missing)}); '
            'nothing to stamp.',
        )

    config_path = root / CONFIG_FILENAME
    if config_path.is_file():
        return config_path, False

    RootConfig.create().save(config_path)
    return config_path, True
