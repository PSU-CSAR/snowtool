"""The cached data-read surface over a catalog :class:`SnowDb`.

:class:`SnowDbReader` *has* a :class:`~snowtool.snowdb.db.SnowDb` (the immutable
catalog, reachable as :attr:`db`) and owns the one piece of read-path state the
catalog deliberately lacks: the :class:`~snowtool.snowdb.raster.tiff_cache.TiffCache`
shared across all of a database's COG reads. :meth:`zonal_stats` -- the sole
consumer of that cache -- lives here, so the cache lives in exactly one type and
test isolation is a type-level fact (a fresh reader is a fresh cache).

The split is by lifecycle, symmetric with ``SnowDbManager``-wraps-``SnowDb``
(siblings over the same catalog, not nested). The catalog is loop-agnostic and
buildable anywhere; the reader's cache is loop-affine (``alru_cache`` binds to the
event loop that first awaits it), so a reader is built inside the loop that will
use it -- the API at app-lifespan scope, the CLI inside its ``asyncio.run``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from snowtool.exceptions import QueryParameterError

# Imported at runtime (not under TYPE_CHECKING) so the ``__provide__`` DI recipe's
# type hints resolve: gazebo reads them to wire the app-scoped provider.
from snowtool.settings import Settings
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.raster.tiff_cache import TiffCache

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from snowtool import types
    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.query import DateQuery
    from snowtool.snowdb.variables import DatasetVariable
    from snowtool.snowdb.zonal_stats import ZonalStats, ZoneSelection


class SnowDbReader:
    """Cached zonal-stats reads over a held catalog :class:`SnowDb`.

    Built around an already-constructed catalog (reachable as :attr:`db`); the
    ``cache`` defaults to a fresh :class:`TiffCache` so a reader can be built with
    nothing but a catalog. The catalog-side reads (coverage guard, AOI raster
    load) stay on :attr:`db`; this surface only adds the cache and the crossed
    reduction that consumes it.
    """

    def __init__(self: Self, db: SnowDb, cache: TiffCache | None = None) -> None:
        self.db = db
        self.cache = cache if cache is not None else TiffCache()

    @classmethod
    def __provide__(cls: type[Self], db: SnowDb, settings: Settings) -> Self:
        """DI recipe (gazebo): build the app-scoped reader from the catalog + settings.

        Registered as ``providers.app(SnowDbReader)``; gazebo resolves ``db`` and
        ``settings`` from their app-scoped bindings and builds this inside the app's
        event loop at lifespan, so the loop-affine cache is born in that loop. The
        cache is sized from ``settings.tiff_cache_size``.
        """
        return cls(db, TiffCache(settings.tiff_cache_size))

    @staticmethod
    def _resolve_variables(
        dataset: Dataset,
        variable_keys: Iterable[str] | None,
    ) -> set[DatasetVariable]:
        """The :class:`DatasetVariable`\\ s named by ``variable_keys``.

        ``None`` (or an empty selection) means every variable the dataset defines;
        an unknown key raises a clean ``ValueError`` listing the choices.
        """
        available = dataset.spec.variables
        keys = None if variable_keys is None else list(variable_keys)
        if not keys:
            return set(available.values())
        resolved: set[DatasetVariable] = set()
        for key in keys:
            try:
                resolved.add(available[key])
            except KeyError as e:
                raise QueryParameterError(
                    f'Unknown variable {key!r} for dataset {dataset.spec.name!r}; '
                    f'available: {", ".join(sorted(available))}.',
                ) from e
        return resolved

    async def zonal_stats(
        self: Self,
        triplet: types.StationTriplet,
        dataset_name: str,
        query: DateQuery,
        *,
        variable_keys: Iterable[str] | None = None,
        zone_selections: Sequence[ZoneSelection] = (),
        allow_partial: bool = False,
        max_zone_cells: int | None = None,
    ) -> ZonalStats:
        """Compute zonal statistics for one AOI over one dataset.

        The shared read seam behind the ``query stats`` CLI command and the HTTP
        stats routes: it guards coverage, loads the burned AOI raster, resolves the
        requested variables, builds the raster collection for ``query``, and runs
        the crossed-zone reduction over the reader's cache. ``variable_keys``
        defaults to every variable the dataset defines; ``zone_selections`` defaults
        to none (a whole-basin reduction). Raises a clean error when the
        dataset/variable is unknown, the pourpoint is not covered
        (:class:`~snowtool.exceptions.PourpointCoverageError`), or the AOI raster
        has not been rasterized (:class:`FileNotFoundError`).
        """
        from snowtool.snowdb.raster.collection import RasterCollection
        from snowtool.snowdb.zonal_stats import DEFAULT_MAX_ZONE_CELLS, ZonalStats

        dataset = self.db.datasets[dataset_name]
        # Refuse a silently-clipped result: the AOI must be inside the dataset's
        # served footprint (fully, unless allow_partial), checked before any read.
        self.db.require_pourpoint_coverage(
            triplet,
            dataset_name,
            allow_partial=allow_partial,
        )

        variables = self._resolve_variables(dataset, variable_keys)
        aoi_raster = dataset.load_aoi_raster(triplet)
        collection = RasterCollection.from_variables_query(query, variables, dataset)
        return await ZonalStats.calculate(
            aoi_raster,
            collection,
            self.cache,
            dataset,
            zone_selections,
            max_zone_cells=(
                DEFAULT_MAX_ZONE_CELLS if max_zone_cells is None else max_zone_cells
            ),
        )
