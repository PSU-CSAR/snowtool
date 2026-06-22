"""The built-in zone-layer provider registry.

Mirrors :mod:`snowtool.snowdb.datasets`'s ``DEFAULT_DATASET_SPECS``: the providers
a :class:`~snowtool.snowdb.db.SnowDb` builds and reads zone layers with, passed in
(not a global) so tests/entrypoints can supply their own. Adding a zone-layer kind
is one provider plus one entry here.
"""

from __future__ import annotations

from snowtool.snowdb.landcover import LandCoverProvider
from snowtool.snowdb.terrain import TerrainProvider

DEFAULT_ZONE_LAYER_PROVIDERS = (TerrainProvider(), LandCoverProvider())
