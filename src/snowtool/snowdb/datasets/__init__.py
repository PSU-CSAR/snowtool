"""The built-in dataset definitions and the dataset registry.

Each dataset kind lives in its own module here (:mod:`.snodas`, :mod:`.swann`,
...), holding that dataset's variables, grid :class:`~snowtool.snowdb.spec.DatasetSpec`,
and :class:`~snowtool.snowdb.ingest.Ingester`. ``DEFAULT_DATASET_SPECS`` collects
them; a :class:`~snowtool.snowdb.db.SnowDb` is built from a
:class:`~snowtool.snowdb.config.RootConfig`, so ``DEFAULT_DATASET_SPECS`` backs
the dataset templates (below) and the tests rather than being passed to a SnowDb
directly. The public names are re-exported here so callers import
``from snowtool.snowdb.datasets import ...`` regardless of which per-dataset
module defines them.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from snowtool.snowdb.config import DatasetConfig
from snowtool.snowdb.spec import DatasetSpec

from .instarr import INSTARR_SPEC, INSTARR_VARIABLES, InstarrIngester
from .snodas import SNODAS_SPEC, SNODAS_VARIABLES, Product, SnodasIngester
from .swann import SWANN_800M_SPEC, SWANN_800M_VARIABLES, SwannIngester

if TYPE_CHECKING:
    from snowtool.snowdb.ingest import Ingester

DEFAULT_DATASET_SPECS: tuple[DatasetSpec, ...] = (
    SNODAS_SPEC,
    SWANN_800M_SPEC,
    INSTARR_SPEC,
)

# The ingester registry: a dataset config names its ingester by one of these keys
# (its ``kind``) and the ingest path resolves the code from here (reads/queries
# never touch it). Note the key is the *kind* (``swann``), distinct from a dataset
# *name* (``swann-800m``). Built from the SAME ingester instances the built-in
# specs embed (keyed by each ingester's ``kind``), so the registry and the specs
# never drift into two instances of a kind.
INGESTERS: dict[str, Ingester] = {
    spec.ingester.kind: spec.ingester
    for spec in DEFAULT_DATASET_SPECS
    if spec.ingester is not None
}


def config_from_spec(spec: DatasetSpec) -> DatasetConfig:
    """Produce the self-describing :class:`DatasetConfig` for a built-in spec.

    The inverse of :meth:`DatasetSpec.from_config`; round-tripping a built-in spec
    through this and back reproduces the spec exactly -- the guarantee behind the
    templates below.
    """
    return DatasetConfig(
        grid=spec.grid_params,
        variables=dict(spec.variables),
        # The ingester names its own registry key (its ``kind``); None for a
        # read-only/derived spec with no ingester.
        ingester=spec.ingester.kind if spec.ingester is not None else None,
        zones=spec.zones,
        footprint=spec.footprint,
    )


# Canned dataset configs (keyed by dataset name) stamped by ``dataset create
# --template``. Derived from the built-in specs so a dataset kind's definition
# stays in one place -- e.g. INSTARR's grid + footprint keep their readable
# arithmetic in datasets/instarr.py and the template is produced from them.
DATASET_TEMPLATES: dict[str, DatasetConfig] = {
    spec.name: config_from_spec(spec) for spec in DEFAULT_DATASET_SPECS
}

# Packaged per-template nodata masks (template name -> filename in this
# package). A template listed here stamps its mask into the created dataset
# (`dataset create` copies it beside the dataset config and points the config's
# `nodata_mask` at it). Masks are package data, not spec fields: a spec is the
# path-independent definition, while a mask is a shipped artifact. Private:
# `template_nodata_mask` is the lookup API.
_DATASET_TEMPLATE_MASKS: dict[str, str] = {
    'snodas': 'snodas-nodata-mask.tif',
    'swann-800m': 'swann-nodata-mask.tif',
}


def template_nodata_mask(name: str) -> Path | None:
    """The packaged nodata-mask file for template ``name``, or ``None``.

    Resolved through ``importlib.resources`` so it works from any install
    layout; snowtool installs as a normal (unzipped) package, so the resource
    is always a real filesystem path.
    """
    filename = _DATASET_TEMPLATE_MASKS.get(name)
    if filename is None:
        return None
    return Path(str(resources.files(__package__) / filename))


__all__ = [
    'DATASET_TEMPLATES',
    'DEFAULT_DATASET_SPECS',
    'INGESTERS',
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
    'config_from_spec',
    'template_nodata_mask',
]
