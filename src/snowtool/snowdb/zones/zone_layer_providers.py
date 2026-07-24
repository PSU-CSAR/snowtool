"""The built-in zone-layer provider registry.

Mirrors :mod:`snowtool.snowdb.datasets`'s ``DEFAULT_DATASET_SPECS``: the providers
a :class:`~snowtool.snowdb.db.SnowDb` builds and reads zone layers with, passed in
(not a global) so tests/entrypoints can supply their own. Adding a zone-layer kind
is one provider factory plus one entry here.
"""

from __future__ import annotations

from snowtool.snowdb.zones.landcover import landcover_provider
from snowtool.snowdb.zones.terrain import terrain_provider

DEFAULT_ZONE_LAYER_PROVIDERS = (terrain_provider(), landcover_provider())
