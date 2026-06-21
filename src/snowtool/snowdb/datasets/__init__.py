"""The built-in dataset definitions and the dataset registry.

Each dataset kind lives in its own module here (:mod:`.snodas`, :mod:`.swann`,
...), holding that dataset's variables, grid :class:`~snowtool.snowdb.spec.DatasetSpec`,
and :class:`~snowtool.snowdb.ingest.Ingester`. ``DEFAULT_DATASET_SPECS`` collects
them; it is what the app/CLI pass to a :class:`~snowtool.snowdb.db.SnowDb` (tests
may pass a subset or their own synthetic specs). The public names are re-exported
here so callers import ``from snowtool.snowdb.datasets import ...`` regardless of
which per-dataset module defines them.
"""

from __future__ import annotations

from snowtool.snowdb.spec import DatasetSpec

from .instarr import INSTARR_SPEC, INSTARR_VARIABLES, InstarrIngester
from .snodas import SNODAS_SPEC, SNODAS_VARIABLES, Product, SnodasIngester
from .swann import SWANN_800M_SPEC, SWANN_800M_VARIABLES, SwannIngester

# The built-in datasets; the app/CLI pass this to SnowDb.
DEFAULT_DATASET_SPECS: tuple[DatasetSpec, ...] = (
    SNODAS_SPEC,
    SWANN_800M_SPEC,
    INSTARR_SPEC,
)

__all__ = [
    'DEFAULT_DATASET_SPECS',
    'INSTARR_SPEC',
    'INSTARR_VARIABLES',
    'SNODAS_SPEC',
    'SNODAS_VARIABLES',
    'SWANN_800M_SPEC',
    'SWANN_800M_VARIABLES',
    'InstarrIngester',
    'Product',
    'SnodasIngester',
    'SwannIngester',
]
