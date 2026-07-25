"""Unit tests for the typed validation Issue vocabulary."""

import pytest

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
