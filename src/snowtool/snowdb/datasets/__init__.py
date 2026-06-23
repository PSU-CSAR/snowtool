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

from dataclasses import asdict
from typing import TYPE_CHECKING

import shapely

from snowtool.snowdb.config import (
    DatasetConfig,
    GridConfig,
    UnitConfig,
    VariableConfig,
)
from snowtool.snowdb.spec import DatasetSpec

from .instarr import INSTARR_SPEC, INSTARR_VARIABLES, InstarrIngester
from .snodas import SNODAS_SPEC, SNODAS_VARIABLES, Product, SnodasIngester
from .swann import SWANN_800M_SPEC, SWANN_800M_VARIABLES, SwannIngester

if TYPE_CHECKING:
    from snowtool.snowdb.ingest import Ingester

# The built-in datasets; the app/CLI pass this to SnowDb.
DEFAULT_DATASET_SPECS: tuple[DatasetSpec, ...] = (
    SNODAS_SPEC,
    SWANN_800M_SPEC,
    INSTARR_SPEC,
)

# The ingester registry: a dataset config names its ingester by one of these keys
# and the ingest path resolves the code from here (reads/queries never touch it).
# Specific ingesters only -- no generic/parameterized ones. Note the key is the
# *kind* (``swann``), distinct from a dataset *name* (``swann-800m``).
INGESTERS: dict[str, Ingester] = {
    'snodas': SnodasIngester(),
    'swann': SwannIngester(),
    'instarr': InstarrIngester(),
}

# Reverse map for building a config from a spec: each ingester kind is one class,
# so its type identifies its registry name.
_INGESTER_NAME_BY_TYPE: dict[type, str] = {
    type(ingester): name for name, ingester in INGESTERS.items()
}


def config_from_spec(spec: DatasetSpec) -> DatasetConfig:
    """Produce the self-describing :class:`DatasetConfig` for a built-in spec.

    The serialization half of :meth:`DatasetSpec.from_config`: it flattens a
    spec's grid, variables, ``zones``, ingester (by registry name) and
    ``footprint`` (as a GeoJSON geometry mapping) into a config. Round-tripping a
    built-in spec through this and back must reproduce the spec exactly -- the
    guarantee behind the byte-equal templates below.
    """
    ingester_name = (
        _INGESTER_NAME_BY_TYPE[type(spec.ingester)]
        if spec.ingester is not None
        else None
    )
    return DatasetConfig(
        grid=GridConfig(**asdict(spec.grid_params)),
        variables={
            key: VariableConfig(
                unit=UnitConfig(
                    name=variable.unit.name,
                    scale_factor=variable.unit.scale_factor,
                ),
                reducer=variable.reducer,
                dtype=variable.dtype,
                nodata=variable.nodata,
                glob=variable.glob,
            )
            for key, variable in spec.variables.items()
        },
        ingester=ingester_name,
        zones=spec.zones,
        footprint=(
            shapely.geometry.mapping(spec.footprint)
            if spec.footprint is not None
            else None
        ),
    )


# Canned dataset configs (keyed by dataset name) stamped by ``dataset create
# --template``. Derived from the built-in specs so a dataset kind's definition
# stays in one place -- e.g. INSTARR's grid + footprint keep their readable
# arithmetic in datasets/instarr.py and the template is produced from them.
DATASET_TEMPLATES: dict[str, DatasetConfig] = {
    spec.name: config_from_spec(spec) for spec in DEFAULT_DATASET_SPECS
}

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
]
