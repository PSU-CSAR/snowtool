"""Versioned provenance tags: round-tripping and parsing ``v{version}:{digest}``."""

import pytest

from snowtool.snowdb.provenance import parse_format_version, versioned_hash


def test_versioned_hash_formats_version_and_digest():
    assert versioned_hash(3, 'abc123') == 'v3:abc123'


@pytest.mark.parametrize(
    ('versioned', 'expected'),
    [
        (None, None),
        ('', None),
        ('abc123', None),  # unversioned/legacy hash, no 'v' prefix at all
        ('v', None),  # 'v' prefix but no ':' separator
        ('v:abc123', None),  # ':' present but no int between 'v' and ':'
        ('vX:abc123', None),  # non-integer version
        ('version1:abc123', None),  # starts with 'v' but not the 'vN:' shape
        ('v1:abc123', 1),
        ('v42:deadbeef', 42),
        ('v0:abc123', 0),
        ('v-1:abc123', -1),  # parses; the caller decides whether negative is valid
    ],
    ids=[
        'none',
        'empty_string',
        'unversioned_legacy_hash',
        'v_with_no_separator',
        'v_colon_no_digits',
        'non_integer_version',
        'v_prefix_wrong_shape',
        'valid_v1',
        'valid_v42',
        'valid_v0',
        'valid_negative',
    ],
)
def test_parse_format_version(versioned, expected):
    assert parse_format_version(versioned) == expected


def test_versioned_hash_round_trips_through_parse_format_version():
    assert parse_format_version(versioned_hash(7, 'somehash')) == 7
