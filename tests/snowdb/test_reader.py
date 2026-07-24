"""Integration tests for :class:`SnowDbReader` over the synthetic pipeline.

The reader is the cached read seam that ``zonal_stats`` lives on after the
catalog/reader split. These walk it end-to-end over a fully-populated synthetic
snowdb -- a catalog with a stored AOI, generated terrain + land cover, a burned
AOI raster, and an ingested COG -- asserting hand-computable values rather than
just a non-error. The catalog is shareable (immutable); each test builds its own
function-scoped reader (the sole cache owner), so cache isolation is structural.
"""

import asyncio
import logging
import re

from datetime import date

import pytest

from snowtool.exceptions import PourpointCoverageError, QueryParameterError
from snowtool.snowdb.query import DateRangeQuery
from snowtool.snowdb.raster.tiff_cache import TiffCache
from snowtool.snowdb.reader import SnowDbReader
from snowtool.snowdb.zonal_stats import ZoneSelection

from ..conftest import SWE_VALUE, populate_synthetic_root

TRIPLET = '12345:MT:USGS'
# The synthetic snowdb ingests a single date; a closed one-day range selects it.
QUERY = DateRangeQuery(start_date=date(2018, 4, 27), end_date=date(2018, 4, 27))


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
    compact = stats.dump_compact()
    # No zone selection -> whole basin: no zone axes, one cell with an empty zone.
    assert compact.zone_layers == []
    (zone,) = compact.zones
    assert zone.zone == []
    assert zone.area_m2 > 0
    (matrix,) = compact.results.values()
    assert matrix == [[pytest.approx(SWE_VALUE)]]


def test_reader_zone_tokens_parse_to_same_stats_as_zone_selections(reader):
    # A string token is the CLI/HTTP entry point; it must reduce to the exact
    # same stats as passing the equivalent ZoneSelection directly -- both are
    # valid elements of the same `zones` sequence.
    from_tokens = asyncio.run(
        reader.zonal_stats(
            TRIPLET,
            'test',
            QUERY,
            variable_keys=['swe'],
            zones=['terrain.elevation:band_step_ft=500'],
        ),
    )
    from_selections = asyncio.run(
        SnowDbReader(reader.db).zonal_stats(
            TRIPLET,
            'test',
            QUERY,
            variable_keys=['swe'],
            zones=[ZoneSelection('terrain.elevation', 500)],
        ),
    )
    assert from_tokens.dump_compact() == from_selections.dump_compact()


def test_reader_zone_tokens_unknown_layer_raises_query_parameter_error(reader):
    with pytest.raises(QueryParameterError, match='Unknown zone layer'):
        asyncio.run(
            reader.zonal_stats(
                TRIPLET,
                'test',
                QUERY,
                variable_keys=['swe'],
                zones=['nope.nope'],
            ),
        )


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


def test_reader_empty_variable_selection_is_rejected(reader):
    # The "never implicit" invariant lives in the core, not just at the CLI/HTTP
    # boundaries: an empty selection raises rather than silently reading every
    # (expensive) variable.
    with pytest.raises(QueryParameterError, match='At least one variable'):
        asyncio.run(
            reader.zonal_stats(TRIPLET, 'test', QUERY, variable_keys=[]),
        )


def test_reader_holds_max_zone_cells_from_construction(catalog):
    from snowtool.snowdb.zonal_stats import DEFAULT_MAX_ZONE_CELLS

    assert SnowDbReader(catalog).max_zone_cells == DEFAULT_MAX_ZONE_CELLS
    assert SnowDbReader(catalog, max_zone_cells=7).max_zone_cells == 7


def test_reader_applies_its_own_max_zone_cells_cap(catalog):
    # The cap lives on the reader (not passed per query): a reader built with a tiny
    # cap rejects a crossed query whose zone product exceeds it, before any raster read.
    from snowtool.snowdb.zonal_stats import ZoneSelection

    reader = SnowDbReader(catalog, max_zone_cells=4)
    with pytest.raises(ValueError, match='max_zone_cells'):
        asyncio.run(
            reader.zonal_stats(
                TRIPLET,
                'test',
                QUERY,
                variable_keys=['swe'],
                zones=[ZoneSelection('terrain.elevation')],
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
    with pytest.raises(FileNotFoundError, match='pourpoint rasterize'):
        asyncio.run(reader.zonal_stats(TRIPLET, 'test', QUERY, variable_keys=['swe']))


def test_reader_coverage_guard_rejects_unknown_aoi(reader):
    # The guard runs before any read: an AOI with no stored record cannot be covered.
    with pytest.raises((FileNotFoundError, PourpointCoverageError)):
        asyncio.run(
            reader.zonal_stats('00000:MT:USGS', 'test', QUERY, variable_keys=['swe']),
        )


def _log_fields(record: logging.LogRecord) -> dict[str, str]:
    """Parse the ``key=value`` pairs out of a formatted log message."""
    return dict(re.findall(r'(\w+)=(\S+)', record.getMessage()))


@pytest.mark.parametrize(
    ('zones', 'expected_zone_axes', 'expected_cells'),
    [
        ((), 0, 1),  # whole-basin: no axes, one (empty-zone) cell.
        pytest.param(
            'terrain_elevation',
            1,
            16,
            id='single-axis-16-elevation-bands',
        ),
    ],
)
def test_reader_logs_one_structured_line_per_query(
    caplog,
    reader,
    zones,
    expected_zone_axes,
    expected_cells,
):
    if zones == 'terrain_elevation':
        zones = [ZoneSelection('terrain.elevation')]

    caplog.set_level(logging.INFO, logger='snowtool.snowdb.reader')
    asyncio.run(
        reader.zonal_stats(
            TRIPLET,
            'test',
            QUERY,
            variable_keys=['swe'],
            zones=zones,
        ),
    )

    records = [r for r in caplog.records if r.name == 'snowtool.snowdb.reader']
    assert len(records) == 1
    fields = _log_fields(records[0])

    assert fields['dataset'] == 'test'
    assert fields['triplet'] == TRIPLET
    assert int(fields['dates']) == 1
    assert int(fields['rasters']) == 1
    assert int(fields['variables']) == 1
    assert int(fields['zone_axes']) == expected_zone_axes
    assert int(fields['cells']) == expected_cells
    assert fields['coverage'] == 'full'
    assert fields['allow_partial'] == 'False'
    assert int(fields['cache_hits']) >= 0
    assert int(fields['cache_misses']) >= 0
    assert float(fields['duration_ms']) > 0


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
