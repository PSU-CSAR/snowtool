"""Shared gazebo dependency-injection aliases for the FastAPI routers.

Both are app-scoped constant providers registered without a ``__provide__``
recipe, so injection is opt-in via the :class:`~gazebo.ext.fastapi.Inject`
marker rather than auto-detected. One definition each, imported by every router
that needs them, rather than a copy per router.
"""

from __future__ import annotations

from typing import Annotated

from gazebo.ext.fastapi import Inject

from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.reader import SnowDbReader

# SnowDb is registered as an app-scoped constant provider (no __provide__ recipe),
# so injection is opt-in via the Inject marker rather than auto-detected.
CatalogDb = Annotated[SnowDb, Inject]

# SnowDbReader is an app-scoped provider without a __provide__ recipe (its recipe is
# supplied in app.py), so injection is opt-in via the Inject marker. It already
# carries its max_zone_cells cap (sized from settings there), so the routes need no
# Settings.
ReaderDep = Annotated[SnowDbReader, Inject]
