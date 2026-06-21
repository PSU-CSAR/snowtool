"""Re-export the synthetic-grid constants for tests that import them relatively.

The fixtures themselves now live in the top-level ``tests/conftest.py`` so the
``cli`` suite can reuse them; this keeps ``from .conftest import ...`` working for
the snowdb tests (e.g. test_pipeline) that pull these constants.
"""

from ..conftest import (  # noqa: F401
    DEM_ELEVATION_M,
    DEM_NODATA,
    ORIGIN_X,
    ORIGIN_Y,
    PX,
    SIZE,
    SWE_VALUE,
    TILE,
    snodas_swe_name,
)
