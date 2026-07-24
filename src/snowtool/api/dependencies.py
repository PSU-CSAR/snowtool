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

CatalogDb = Annotated[SnowDb, Inject]
ReaderDep = Annotated[SnowDbReader, Inject]
