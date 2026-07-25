"""Unit tests for the typed validation Issue vocabulary."""

import pytest

from affine import Affine

from snowtool.snowdb import issues


@pytest.mark.parametrize(
    ('issue', 'expected_message', 'actionable'),
    [
        (issues.NoCoverage(), 'no coverage', False),
        (issues.PartialCoverage(), 'partial coverage', False),
        (issues.NoRaster(), 'no raster', True),
        (issues.OrphanArtifact(), 'orphan raster', False),
        (
            issues.EmptyArtifact(),
            'empty AOI raster (covers no in-grid cells: off-grid or masked)',
            False,
        ),
        (issues.MissingArtifact(), 'missing', True),
        (issues.Unreadable('boom'), 'unreadable: boom', False),
        (
            issues.MissingProvenanceTag('SNOWTOOL_TILE_BBOX'),
            'missing SNOWTOOL_TILE_BBOX tag '
            '(rebuild with `pourpoint rasterize --rebuild`)',
            True,
        ),
        (
            issues.FormatStale(stored=2, current=3),
            'stale format (stored 2 != current 3)',
            True,
        ),
        (
            issues.FormatStale(stored=None, current=3),
            'stale format (stored None != current 3)',
            True,
        ),
        (
            issues.UnverifiableFreshness('source absent'),
            'freshness unverifiable: source absent',
            False,
        ),
    ],
)
def test_issue_message_and_actionability(issue, expected_message, actionable):
    assert issue.message == expected_message
    assert str(issue) == expected_message
    assert issue.actionable is actionable


def test_grid_and_shape_mismatch_messages():
    g = issues.GridMismatch(declared=(0.1, 0.0, -124.7), actual=(0.1, 0.0, -124.73))
    assert 'declared grid transform' in g.message
    assert '(0.1, 0.0, -124.7)' in g.message
    assert '(0.1, 0.0, -124.73)' in g.message
    assert g.actionable is True
    s = issues.ShapeMismatch(declared=(3351, 6935), actual=(256, 256))
    assert '6935' in s.message
    assert '256x256' in s.message
    assert s.actionable is True


def test_render_joins_messages_in_order():
    joined = issues.render([issues.NoRaster(), issues.PartialCoverage()])
    assert joined == 'no raster; partial coverage'
    assert issues.render([]) == ''


def test_grid_issues_clean_when_matching():
    t = Affine(0.00833333, 0.0, -124.73375, 0.0, -0.00833333, 52.8745833)
    assert (
        issues.grid_issues(
            declared_transform=t,
            actual_transform=t,
            declared_shape=(3351, 6935),
            actual_shape=(3351, 6935),
        )
        == []
    )


def test_grid_issues_flags_transform_and_shape():
    declared = Affine(0.00833333, 0.0, -124.73375, 0.0, -0.00833333, 52.8745833)
    actual = Affine(0.00833333, 0.0, -124.73333, 0.0, -0.00833333, 52.875)
    result = issues.grid_issues(
        declared_transform=declared,
        actual_transform=actual,
        declared_shape=(3351, 6935),
        actual_shape=(256, 256),
    )
    types = {type(i) for i in result}
    assert issues.ShapeMismatch in types
    assert issues.GridMismatch in types


def test_grid_issues_tolerates_float_noise():
    a = Affine(
        0.008333333333333,
        0.0,
        -124.73375,
        0.0,
        -0.008333333333333,
        52.8745833333333,
    )
    b = Affine(
        0.008333333333333,
        0.0,
        -124.7337499999950,
        0.0,
        -0.008333333333333,
        52.8745833333312,
    )
    assert (
        issues.grid_issues(
            declared_transform=a,
            actual_transform=b,
            declared_shape=(3351, 6935),
            actual_shape=(3351, 6935),
        )
        == []
    )
