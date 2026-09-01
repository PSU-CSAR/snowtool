import json

import pytest

from snowtool.cli import cli
from snowtool.snowdb.manager import SnowDbManager

from ..conftest import EXPECTED_SNOW_LINE_FT, populate_bound_root

TRIPLET = '12345:MT:USGS'
DATE = '20180427'


@pytest.fixture
def gradient_root(initialized_root, spec, gradient_pourpoint_geojson):
    manager = SnowDbManager.open(initialized_root)
    populate_bound_root(
        manager,
        spec,
        gradient_pourpoint_geojson,
        terrain_gradient=True,
    )
    return initialized_root


@pytest.fixture
def uniform_root(initialized_root, spec, pourpoint_geojson):
    """A populated root with uniform terrain -- one elevation band only."""
    manager = SnowDbManager.open(initialized_root)
    populate_bound_root(manager, spec, pourpoint_geojson)
    return initialized_root


def test_help_happy_path(runner, cli_obj):
    result = runner.invoke(
        cli,
        [
            'snowline',
            '--help',
        ],
        obj=cli_obj,
    )
    assert result.exit_code == 0, result.output


def test_snowline_happy_path(runner, cli_obj, gradient_root):
    result = runner.invoke(
        cli,
        [
            'snowline',
            'test',
            TRIPLET,
            '--dates',
            f'{DATE}/{DATE}',
            '--variable',
            'swe',
            '--band-step-ft',
            '500',
            '--threshold',
            '50',
        ],
        obj=cli_obj,
    )
    assert result.exit_code == 0, result.output
    (line,) = json.loads(result.output).values()
    assert line['elevation_ft'] == pytest.approx(EXPECTED_SNOW_LINE_FT, abs=1.0)
    assert line['lower_band_area_m2'] > 0
    assert line['higher_band_area_m2'] > 0
    assert line['interpolation_span_ft'] > 0


def test_default_opts_applied(runner, cli_obj, gradient_root):
    result = runner.invoke(
        cli,
        [
            'snowline',
            'test',
            TRIPLET,
            '--dates',
            f'{DATE}/{DATE}',
            '--variable',
            'swe',
        ],
        obj=cli_obj,
    )
    assert result.exit_code == 0, result.output
    (line,) = json.loads(result.output).values()
    assert line['elevation_ft'] == pytest.approx(EXPECTED_SNOW_LINE_FT, abs=1.0)
    assert line['lower_band_area_m2'] > 0
    assert line['higher_band_area_m2'] > 0
    assert line['interpolation_span_ft'] > 0


def test_domain_errors_on_uniform_terrain(runner, cli_obj, uniform_root):
    result = runner.invoke(
        cli,
        [
            'snowline',
            'test',
            TRIPLET,
            '--dates',
            f'{DATE}/{DATE}',
            '--variable',
            'swe',
        ],
        obj=cli_obj,
    )
    assert result.exit_code != 0, result.output
    assert 'Traceback' not in result.output
