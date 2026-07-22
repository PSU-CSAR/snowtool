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

import logging
import time

from typing import TYPE_CHECKING, Self

from snowtool.exceptions import QueryParameterError
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.raster.collection import RasterCollection
from snowtool.snowdb.raster.tiff_cache import TiffCache
from snowtool.snowdb.zonal_stats import (
    DEFAULT_MAX_CONCURRENT_RASTERS,
    DEFAULT_MAX_ZONE_CELLS,
    ZonalStats,
    ZoneSelection,
    parse_zone_selection,
)
from snowtool.snowdb.zones.zone_layer import available_zones

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from snowtool import types
    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.query import DateQuery
    from snowtool.snowdb.variables import DatasetVariable

logger = logging.getLogger(__name__)


class SnowDbReader:
    """Cached zonal-stats reads over a held catalog :class:`SnowDb`.

    Built around an already-constructed catalog (reachable as :attr:`db`); the
    ``cache`` defaults to a fresh :class:`TiffCache` so a reader can be built with
    nothing but a catalog. The catalog-side reads (coverage guard, AOI raster
    load) stay on :attr:`db`; this surface adds the two read-path knobs -- the COG
    ``cache`` and the crossed-stats ``max_zone_cells`` cap -- and the crossed
    reduction that consumes them. Both are passed in (the API sizes them from its
    ``Settings``; the CLI/tests take the defaults); the reader imports no settings.
    """

    def __init__(
        self: Self,
        db: SnowDb,
        cache: TiffCache | None = None,
        max_zone_cells: int | None = None,
        max_concurrent_rasters: int | None = None,
    ) -> None:
        self.db = db
        self.cache = cache if cache is not None else TiffCache()
        # Output-size guard for a crossed query (product of the selected zone axes);
        # a read-path cap held here like the cache, not threaded per query.
        self.max_zone_cells = (
            DEFAULT_MAX_ZONE_CELLS if max_zone_cells is None else max_zone_cells
        )
        # Fan-out cap on concurrent per-raster reductions (transient window
        # allocations + unbounded fetch batches); a read-path knob beside the cache,
        # not threaded per query. Bounds peak memory only, never results.
        self.max_concurrent_rasters = (
            DEFAULT_MAX_CONCURRENT_RASTERS
            if max_concurrent_rasters is None
            else max_concurrent_rasters
        )

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
        zones: Sequence[ZoneSelection | str] = (),
        allow_partial: bool = False,
    ) -> ZonalStats:
        """Compute zonal statistics for one AOI over one dataset.

        The shared read seam behind the ``stats`` CLI command and the HTTP
        stats routes: it guards coverage, loads the burned AOI raster, resolves the
        requested variables, builds the raster collection for ``query``, and runs
        the crossed-zone reduction over the reader's cache. ``variable_keys``
        defaults to every variable the dataset defines. ``zones`` defaults to none
        (a whole-basin reduction); each element is either an already-resolved
        :class:`~snowtool.snowdb.zonal_stats.ZoneSelection` (the programmatic form)
        or a CLI/HTTP ``LAYER[:PARAM=VALUE]`` string token, parsed here -- building
        the zone registry once via
        :func:`~snowtool.snowdb.zones.zone_layer.available_zones` and mapping each
        string through :func:`~snowtool.snowdb.zonal_stats.parse_zone_selection` --
        so callers never touch the registry themselves. Raises a clean error when
        the dataset/variable is unknown, a zone token is malformed or names an
        unknown layer, the pourpoint is not covered
        (:class:`~snowtool.exceptions.PourpointCoverageError`), or the AOI raster
        has not been rasterized (:class:`FileNotFoundError`).
        """
        dataset = self.db[dataset_name]
        # Refuse a silently-clipped result: the AOI must be inside the dataset's
        # served footprint (fully, unless allow_partial), checked before any read.
        coverage = self.db.require_pourpoint_coverage(
            triplet,
            dataset_name,
            allow_partial=allow_partial,
        )

        # The registry is built once here (only when a string token needs parsing)
        # and used for every token in this query; an already-resolved
        # ZoneSelection passes through untouched.
        registry = None
        zone_selections: list[ZoneSelection] = []
        for zone in zones:
            if isinstance(zone, ZoneSelection):
                zone_selections.append(zone)
                continue
            if registry is None:
                registry = available_zones(dataset.providers.values())
            zone_selections.append(parse_zone_selection(zone, registry))

        variables = self._resolve_variables(dataset, variable_keys)
        aoi_raster = dataset.load_aoi_raster(triplet)
        collection = RasterCollection.from_variables_query(query, variables, dataset)

        cache_before = self.cache.info()
        start = time.perf_counter()
        stats = await ZonalStats.calculate(
            aoi_raster,
            collection,
            self.cache,
            dataset,
            zone_selections,
            max_zone_cells=self.max_zone_cells,
            max_concurrent_rasters=self.max_concurrent_rasters,
        )
        duration_ms = (time.perf_counter() - start) * 1000
        cache_after = self.cache.info()

        logger.info(
            'zonal_stats dataset=%s triplet=%s dates=%d rasters=%d variables=%d '
            'zone_axes=%d cells=%d coverage=%s allow_partial=%s cache_hits=%d '
            'cache_misses=%d duration_ms=%.1f',
            dataset_name,
            triplet,
            len(collection.dates),
            len(collection),
            len(variables),
            len(zone_selections),
            stats.n_cells,
            coverage.value,
            allow_partial,
            cache_after.hits - cache_before.hits,
            cache_after.misses - cache_before.misses,
            duration_ms,
        )

        return stats
