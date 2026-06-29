"""Integration tests for :class:`SnowDbReader` over the synthetic pipeline.

The reader is the cached read seam that ``zonal_stats`` lives on after the
catalog/reader split. These walk it end-to-end over a fully-populated synthetic
snowdb -- a catalog with a stored AOI, generated terrain + land cover, a burned
AOI raster, and an ingested COG -- asserting hand-computable values rather than
just a non-error. The catalog is shareable (immutable); each test builds its own
function-scoped reader (the sole cache owner), so cache isolation is structural.
"""

import asyncio

from datetime import date

import pytest

from snowtool import types
from snowtool.exceptions import PourpointCoverageError
from snowtool.snowdb.reader import SnowDbReader
from snowtool.snowdb.tiff_cache import TiffCache

from ..conftest import SWE_VALUE, populate_synthetic_root

TRIPLET = '12345:MT:USGS'
# The synthetic snowdb ingests a single date; a closed one-day range selects it.
QUERY = types.DateRangeQuery(start_date=date(2018, 4, 27), end_date=date(2018, 4, 27))


@pytest.fixture
def catalog(tmp_path, spec, pourpoint_geojson):
    """A fully-populated catalog SnowDb (immutable; shareable across readers)."""
    return populate_synthetic_root(tmp_path, spec, pourpoint_geojson)


@pytest.fixture
def reader(catalog):
    """A function-scoped reader over the catalog -- the sole owner of its cache."""
    return SnowDbReader(catalog)


def test_reader_whole_basin_zonal_stats(reader):
    stats = asyncio.run(
        reader.zonal_stats(TRIPLET, 'test', QUERY, variable_keys=['swe']),
    )
    dumped = stats.dump()
    assert len(dumped) == 1
    # No zone selection -> whole basin: no zone axes, one cell with an empty zone.
    assert dumped[0].zone_layers == []
    (cell,) = dumped[0].zones
    assert cell.zone == []
    assert cell.mean_swe_mm == SWE_VALUE
    assert cell.area_m2 > 0


def test_reader_unknown_variable_is_clean_error(reader):
    with pytest.raises(ValueError, match='Unknown variable'):
        asyncio.run(
            reader.zonal_stats(
                TRIPLET,
                'test',
                QUERY,
                variable_keys=['nope'],
            ),
        )


def test_reader_missing_aoi_raster_raises(tmp_path, spec, pourpoint_geojson):
    # AOI imported (so coverage passes) but never rasterized -> a clean prereq error.
    catalog = populate_synthetic_root(
        tmp_path,
        spec,
        pourpoint_geojson,
        rasterize=False,
    )
    reader = SnowDbReader(catalog)
    with pytest.raises(FileNotFoundError, match='aoi rasterize'):
        asyncio.run(reader.zonal_stats(TRIPLET, 'test', QUERY, variable_keys=['swe']))


def test_reader_coverage_guard_rejects_unknown_aoi(reader):
    # The guard runs before any read: an AOI with no stored record cannot be covered.
    with pytest.raises((FileNotFoundError, PourpointCoverageError)):
        asyncio.run(
            reader.zonal_stats('00000:MT:USGS', 'test', QUERY),
        )


def test_reader_owns_independent_caches(catalog):
    # The cache is a type-level fact of the reader: each reader is a fresh cache,
    # so sharing a catalog never shares read-path state.
    one = SnowDbReader(catalog)
    two = SnowDbReader(catalog)
    assert isinstance(one.cache, TiffCache)
    assert one.cache is not two.cache
    # An injected cache is honored (the API/CLI size it from settings / the loop).
    injected = TiffCache(maxsize=4)
    assert SnowDbReader(catalog, injected).cache is injected
