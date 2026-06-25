"""CLI tests for the ``query`` group (stats + dates) over the synthetic pipeline.

``query stats`` needs the full set of prerequisites -- a stored AOI, generated
terrain + land cover, a burned AOI raster, and an ingested COG -- so a fixture
lays all of them down on the synthetic ``test`` dataset, then the commands run
against the same root via the injected ``cli_obj`` context.
"""

import json

from datetime import date

import numpy
import pytest

from snowtool.cli import cli
from snowtool.snowdb.aoi import AOI
from snowtool.snowdb.cog import write_cog
from snowtool.snowdb.manager import SnowDbManager

from ..conftest import (
    SIZE,
    SWE_VALUE,
    TILE,
    snodas_swe_name,
    write_landcover,
    write_terrain,
)

TRIPLET = '12345:MT:USGS'
DATE = '20180427'

CSV_HEADER = (
    'date,terrain.elevation_min_ft,terrain.elevation_max_ft,area_m2,mean_swe_mm'
)


def _ingest_swe_cog(dataset) -> None:
    out_dir = dataset.date_dir(date(2018, 4, 27))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_cog(
        out_dir / f'{snodas_swe_name(DATE)}.tif',
        numpy.full((SIZE, SIZE), SWE_VALUE, dtype=numpy.int16),
        transform=dataset.grid.base_grid.transform,
        tile_size=TILE,
        predictor=2,
    )


@pytest.fixture
def populated_root(initialized_root, aoi_geojson):
    """The synthetic root populated end-to-end for a stats query."""
    manager = SnowDbManager.open(initialized_root)
    manager.import_aois(aoi_geojson)
    dataset = manager.db['test']
    write_terrain(dataset)
    write_landcover(dataset)
    dataset.rasterize_aoi(AOI.from_geojson(aoi_geojson), force=True)
    _ingest_swe_cog(dataset)
    return initialized_root


def test_query_stats_whole_basin_json(runner, cli_obj, populated_root):
    result = runner.invoke(
        cli,
        [
            'query',
            'stats',
            TRIPLET,
            '-d',
            'test',
            '--start',
            DATE,
            '--end',
            DATE,
            '--variable',
            'swe',
            '--format',
            'json',
        ],
        obj=cli_obj,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert len(payload) == 1
    # No --zone -> whole basin: no zone axes, one cell with an empty zone list.
    assert payload[0]['zone_layers'] == []
    (cell,) = payload[0]['zones']
    assert cell['zone'] == []
    assert cell['mean_swe_mm'] == SWE_VALUE
    assert cell['area_m2'] > 0


def test_query_stats_csv_with_elevation_zone(runner, cli_obj, populated_root):
    result = runner.invoke(
        cli,
        [
            'query',
            'stats',
            TRIPLET,
            '-d',
            'test',
            '--start',
            DATE,
            '--end',
            DATE,
            '--variable',
            'swe',
            '--zone',
            'terrain.elevation',
            '--format',
            'csv',
        ],
        obj=cli_obj,
    )
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0] == CSV_HEADER
    # 16 elevation bands -> 16 rows; exactly one (3000-4000 ft) carries the SWE.
    rows = lines[1:]
    assert len(rows) == 16
    populated = [r for r in rows if r.endswith(f',{float(SWE_VALUE)}')]
    assert len(populated) == 1
    assert populated[0].startswith('2018-04-27,3000,4000,')


def test_query_stats_threshold_zone_override(runner, cli_obj, populated_root):
    # The synthetic forest layer is 100%; a threshold above 100 flips the basin to
    # "unforested", proving the --zone LAYER:override syntax reaches the scheme.
    result = runner.invoke(
        cli,
        [
            'query',
            'stats',
            TRIPLET,
            '-d',
            'test',
            '--start',
            DATE,
            '--end',
            DATE,
            '--variable',
            'swe',
            '--zone',
            'landcover.forest_cover:100.5',
            '--format',
            'json',
        ],
        obj=cli_obj,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    populated = [c for c in payload[0]['zones'] if c['area_m2'] > 0]
    (cell,) = populated
    (ref,) = cell['zone']
    assert ref['layer'] == 'landcover.forest_cover'
    assert ref['side'] == 'below'
    assert ref['label'] == 'unforested'
    assert ref['threshold'] == 100.5


def test_query_stats_day_of_year_mode(runner, cli_obj, populated_root):
    # The ingested COG is 2018-04-27; a single-year DOY query selects exactly it.
    result = runner.invoke(
        cli,
        [
            'query',
            'stats',
            TRIPLET,
            '-d',
            'test',
            '--variable',
            'swe',
            '--doy',
            '4',
            '27',
            '--start-year',
            '2018',
            '--end-year',
            '2018',
            '--format',
            'json',
        ],
        obj=cli_obj,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    (cell,) = payload[0]['zones']
    assert cell['mean_swe_mm'] == SWE_VALUE


def test_query_stats_rejects_mixing_date_modes(runner, cli_obj, populated_root):
    result = runner.invoke(
        cli,
        [
            'query',
            'stats',
            TRIPLET,
            '-d',
            'test',
            '--variable',
            'swe',
            '--start',
            DATE,
            '--end',
            DATE,
            '--doy',
            '4',
            '27',
        ],
        obj=cli_obj,
    )
    assert result.exit_code != 0
    assert 'not both' in result.output


def test_query_stats_missing_aoi_raster_is_clean_error(
    runner,
    cli_obj,
    initialized_root,
    aoi_geojson,
):
    # AOI imported (so coverage passes) but never rasterized -> a clean prereq error.
    SnowDbManager.open(initialized_root).import_aois(aoi_geojson)
    result = runner.invoke(
        cli,
        [
            'query',
            'stats',
            TRIPLET,
            '-d',
            'test',
            '--start',
            DATE,
            '--end',
            DATE,
            '--variable',
            'swe',
        ],
        obj=cli_obj,
    )
    assert result.exit_code != 0
    assert 'aoi rasterize' in result.output


def test_query_stats_unknown_zone_is_clean_error(runner, cli_obj, populated_root):
    result = runner.invoke(
        cli,
        [
            'query',
            'stats',
            TRIPLET,
            '-d',
            'test',
            '--start',
            DATE,
            '--end',
            DATE,
            '--variable',
            'swe',
            '--zone',
            'terrain.nope',
        ],
        obj=cli_obj,
    )
    assert result.exit_code != 0
    assert 'Unknown zone layer' in result.output


def test_query_stats_requires_a_date_selection(runner, cli_obj, populated_root):
    result = runner.invoke(
        cli,
        ['query', 'stats', TRIPLET, '-d', 'test', '--variable', 'swe'],
        obj=cli_obj,
    )
    assert result.exit_code != 0
    assert 'date range' in result.output


def test_query_stats_unknown_dataset_is_clean_error(runner, cli_obj, populated_root):
    result = runner.invoke(
        cli,
        [
            'query',
            'stats',
            TRIPLET,
            '-d',
            'nope',
            '--start',
            DATE,
            '--end',
            DATE,
            '--variable',
            'swe',
        ],
        obj=cli_obj,
    )
    assert result.exit_code != 0
    assert 'No such dataset' in result.output


def test_query_dates_lists_ingested_dates(runner, cli_obj, populated_root):
    result = runner.invoke(cli, ['query', 'dates', 'test'], obj=cli_obj)
    assert result.exit_code == 0, result.output
    assert '2018-04-27' in result.output


def test_query_dates_filters_by_range(runner, cli_obj, populated_root):
    result = runner.invoke(
        cli,
        ['query', 'dates', 'test', '--start', '20190101'],
        obj=cli_obj,
    )
    assert result.exit_code == 0, result.output
    assert '2018-04-27' not in result.output
