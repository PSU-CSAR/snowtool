"""Per-pourpoint per-dataset coverage: does a dataset's grid contain a basin?

Coverage varies per dataset because each has its own grid/CRS/extent (instarr is
a MODIS-sinusoidal western block; SNODAS/SWANN are geographic national grids), so
a basin fully served by one dataset may be only partially -- or not at all -- inside
another. :func:`dataset_coverage` is the pure kernel that classifies one pourpoint
against a :class:`CoverageDomain` (the region a dataset can serve, in its grid's
CRS) into :class:`Coverage`; it is reprojection-correct (the basin is moved into
the domain's CRS before the containment test) and reads no rasters.

The domain defaults to the full grid-extent rectangle but may instead be a
dataset's declared *footprint* polygon -- the region it actually serves (e.g. a
MODIS block minus a tile that is never populated) -- so a basin over a *static*
nodata hole is not mis-reported as fully covered. Per-date data gaps (clouds, a
missing day's tile) are deliberately a separate, per-result concern, not part of
this static geometric domain.
"""

from __future__ import annotations

import enum

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import shapely

from snowtool.exceptions import PourpointCoverageError

if TYPE_CHECKING:
    from griffine.grid import TiledAffineGrid

    from snowtool.snowdb.pourpoint import Pourpoint


class Coverage(enum.Enum):
    """How fully a dataset's grid contains a pourpoint's basin geometry.

    ``FULL`` -- the grid extent covers the whole basin (the only state a zonal
    query may run over without clipping); ``PARTIAL`` -- the basin overlaps the
    grid but spills outside it (a query would silently use only the in-grid
    portion); ``NONE`` -- the basin is entirely outside the grid (an empty mask).
    """

    FULL = 'full'
    PARTIAL = 'partial'
    NONE = 'none'


def _grid_extent_polygon(grid: TiledAffineGrid) -> shapely.Polygon:
    """The grid's full extent as a rectangle in the grid's *own* CRS.

    Same corners as :func:`~snowtool.snowdb.grid.grid_extent_4326`, but kept in
    the grid CRS (no reprojection): the basin is brought *to* this CRS instead, so
    the containment test is exact even for a projected grid like MODIS sinusoidal.
    """
    base = grid.base_grid
    t = base.transform
    xmin = t.c
    ymax = t.f
    xmax = t.c + base.cols * t.a
    ymin = t.f + base.rows * t.e
    return shapely.box(xmin, ymin, xmax, ymax)


@dataclass(frozen=True)
class CoverageDomain:
    """The region a dataset can serve, as a polygon in its grid's CRS.

    ``polygon`` defaults (via :meth:`from_grid`) to the grid-extent rectangle but
    may instead be a dataset's declared ``footprint`` -- the region it actually
    serves (e.g. a never-ingested MODIS tile left out) -- so coverage reflects the
    dataset's real static domain rather than just its bounding box.
    """

    crs: Any
    polygon: shapely.Geometry

    @classmethod
    def from_grid(
        cls: type[Self],
        grid: TiledAffineGrid,
        *,
        footprint: shapely.Geometry | None = None,
    ) -> Self:
        """Build a domain from a grid, using ``footprint`` if given else the extent.

        ``footprint`` (in the grid's CRS, the same space the extent rectangle is
        built in) *is* the served region when supplied; omitted, the domain
        defaults to the full grid-extent rectangle.
        """
        crs = grid.crs
        if crs is None:  # pragma: no cover - make_grid always sets a CRS
            raise ValueError('grid has no CRS')
        polygon = footprint if footprint is not None else _grid_extent_polygon(grid)
        return cls(crs, polygon)


def dataset_coverage(pourpoint: Pourpoint, domain: CoverageDomain) -> Coverage:
    """Classify how fully ``domain`` contains ``pourpoint``'s basin.

    The basin (stored WGS84) is reprojected into ``domain``'s CRS and tested
    against its polygon. ``covers`` (not ``contains``) is used for ``FULL`` so a
    basin lying exactly on the domain boundary still counts as fully covered.
    """
    geometry = pourpoint.geometry_in_crs(domain.crs)
    if domain.polygon.covers(geometry):
        return Coverage.FULL
    if domain.polygon.intersects(geometry):
        return Coverage.PARTIAL
    return Coverage.NONE


def require_full_coverage(
    coverage: Coverage,
    *,
    triplet: str,
    dataset: str,
    allow_partial: bool = False,
) -> None:
    """Guard a query: raise unless the dataset fully covers the pourpoint.

    ``FULL`` always passes. ``PARTIAL`` passes only when ``allow_partial`` is set
    (the caller knowingly wants the in-grid portion). ``NONE`` always raises --
    an off-grid basin has no pixels, so there is nothing to clip to. Raises
    :class:`~snowtool.exceptions.PourpointCoverageError`.
    """
    if coverage is Coverage.FULL:
        return
    if coverage is Coverage.PARTIAL and allow_partial:
        return
    raise PourpointCoverageError(triplet, dataset, coverage)
