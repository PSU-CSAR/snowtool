"""The pure dataset_coverage kernel: full / partial / none, across CRSs."""

import pytest

from snowtool.exceptions import PourpointCoverageError
from snowtool.snowdb.coverage import (
    Coverage,
    CoverageDomain,
    dataset_coverage,
    require_full_coverage,
)
from snowtool.snowdb.datasets.instarr import INSTARR_SPEC
from snowtool.snowdb.pourpoint import Pourpoint

from ..conftest import synthetic_grid, write_pourpoint_record


def _aoi(tmp_path, polygon, triplet='12345:MT:USGS'):
    """An AOI parsed from a minimal point+polygon pourpoint geojson."""
    path = write_pourpoint_record(
        tmp_path / f'{triplet.replace(":", "_")}.geojson',
        triplet,
        polygon=polygon,
        point=polygon[0],
    )
    return Pourpoint.from_geojson(path)


def _ring(x0, y0, x1, y1):
    """A closed rectangular ring (lon/lat) from two corners."""
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


# A WGS84 domain spanning lon [-120, -114.88], lat [39.88, 45].
def _geographic_domain():
    grid = synthetic_grid(crs=4326)
    return CoverageDomain.from_grid(grid, grid.crs)


# --- matching-CRS (geographic) ----------------------------------------------


def test_full_when_basin_inside_geographic_grid(tmp_path):
    domain = _geographic_domain()
    aoi = _aoi(tmp_path, _ring(-119.9, 44.9, -119.0, 44.0))
    assert dataset_coverage(aoi, domain) is Coverage.FULL


def test_partial_when_basin_straddles_the_grid_edge(tmp_path):
    domain = _geographic_domain()
    # Crosses the western edge (lon -120): part in, part out.
    aoi = _aoi(tmp_path, _ring(-120.5, 44.9, -119.5, 44.0))
    assert dataset_coverage(aoi, domain) is Coverage.PARTIAL


def test_none_when_basin_outside_the_grid(tmp_path):
    domain = _geographic_domain()
    aoi = _aoi(tmp_path, _ring(-110.0, 44.9, -109.0, 44.0))
    assert dataset_coverage(aoi, domain) is Coverage.NONE


def test_full_when_basin_touches_the_grid_boundary(tmp_path):
    # A basin flush against the western edge is covered (covers, not contains):
    # boundary contact must not be misread as partial.
    domain = _geographic_domain()
    aoi = _aoi(tmp_path, _ring(-120.0, 44.9, -119.0, 44.0))
    assert dataset_coverage(aoi, domain) is Coverage.FULL


# --- differing CRS: MODIS sinusoidal (reprojected containment) --------------


def test_full_against_modis_sinusoidal_grid(tmp_path):
    # The live instarr domain (projected, sinusoidal). A western-US basin
    # reprojects into sinusoidal and falls fully inside the h08-10 x v04-05 block.
    domain = INSTARR_SPEC.coverage_domain
    aoi = _aoi(tmp_path, _ring(-119.6, 38.2, -119.4, 38.0))
    assert dataset_coverage(aoi, domain) is Coverage.FULL


def test_none_against_modis_sinusoidal_grid(tmp_path):
    # An eastern-US basin is well outside the western sinusoidal block.
    domain = INSTARR_SPEC.coverage_domain
    aoi = _aoi(tmp_path, _ring(-80.0, 40.0, -79.8, 39.8))
    assert dataset_coverage(aoi, domain) is Coverage.NONE


def test_instarr_excludes_the_empty_h10v05_tile(tmp_path):
    # A basin inside the never-ingested h10v05 corner sits within the grid's
    # bounding rectangle but outside instarr's real domain: full against the bare
    # extent, none once the empty tile is carved out.
    grid = INSTARR_SPEC.grid
    aoi = _aoi(tmp_path, _ring(-91.65, 35.05, -91.55, 34.95))
    domain = CoverageDomain.from_grid(grid, grid.crs)
    assert dataset_coverage(aoi, domain) is Coverage.FULL
    assert dataset_coverage(aoi, INSTARR_SPEC.coverage_domain) is Coverage.NONE


# --- the query guard: require_full_coverage ---------------------------------


def test_guard_passes_when_full():
    # No raise.
    require_full_coverage(Coverage.FULL, triplet='1:MT:USGS', dataset='test')


def test_guard_raises_on_partial_by_default():
    with pytest.raises(PourpointCoverageError, match='partially covered'):
        require_full_coverage(Coverage.PARTIAL, triplet='1:MT:USGS', dataset='test')


def test_guard_allow_partial_bypasses_partial():
    require_full_coverage(
        Coverage.PARTIAL,
        triplet='1:MT:USGS',
        dataset='test',
        allow_partial=True,
    )


def test_guard_raises_on_none_even_with_allow_partial():
    # NONE is never allowed: an off-grid basin has no pixels to clip to.
    with pytest.raises(PourpointCoverageError, match='not covered'):
        require_full_coverage(
            Coverage.NONE,
            triplet='1:MT:USGS',
            dataset='test',
            allow_partial=True,
        )


def test_guard_error_carries_context():
    with pytest.raises(PourpointCoverageError) as exc_info:
        require_full_coverage(Coverage.NONE, triplet='42:CA:USGS', dataset='instarr')
    err = exc_info.value
    assert err.triplet == '42:CA:USGS'
    assert err.dataset == 'instarr'
    assert err.coverage is Coverage.NONE
