"""Remote pourpoint source resolution: URL parsing, flattening, local passthrough.

The HTTP fetch paths are exercised by the ``network``-marked tests in
``test_remote_network.py``; here we pin the pure logic that needs no
network -- URL parsing, the flat-name collision guard, and the local passthrough.
"""

import pytest

from snowtool.cli._remote import (
    GitHubTree,
    _flat_names,
    is_http_url,
    materialize_dir,
    materialize_file,
    parse_github_url,
)
from snowtool.exceptions import RemoteSourceError


@pytest.mark.parametrize(
    ('src', 'expected'),
    [
        ('https://example.com/x.geojson', True),
        ('http://example.com/x.geojson', True),
        ('/local/path/dir', False),
        ('dir', False),
        ('file:///local/x.geojson', False),
    ],
)
def test_is_http_url(src, expected):
    assert is_http_url(src) is expected


@pytest.mark.parametrize(
    ('url', 'expected'),
    [
        (
            'https://github.com/PSU-CSAR/BAGIS-pourpoints/tree/main/reference',
            GitHubTree('PSU-CSAR', 'BAGIS-pourpoints', 'main', 'reference'),
        ),
        # trailing slash on the tree URL
        (
            'https://github.com/O/R/tree/main/reference/',
            GitHubTree('O', 'R', 'main', 'reference'),
        ),
        # a nested subdir keeps every segment after the ref
        (
            'https://github.com/O/R/tree/dev/a/b/c',
            GitHubTree('O', 'R', 'dev', 'a/b/c'),
        ),
        # tree URL with no subdir -> repo root on that ref
        (
            'https://github.com/O/R/tree/main',
            GitHubTree('O', 'R', 'main', ''),
        ),
        # bare repo URL -> default branch (None) at the root
        (
            'https://github.com/O/R',
            GitHubTree('O', 'R', None, ''),
        ),
        # bare repo URL, trailing slash and .git suffix both stripped
        ('https://github.com/O/R.git', GitHubTree('O', 'R', None, '')),
        ('https://github.com/O/R/', GitHubTree('O', 'R', None, '')),
    ],
)
def test_parse_github_url(url, expected):
    assert parse_github_url(url) == expected


@pytest.mark.parametrize(
    'url',
    [
        # a raw single-file URL is not a folder -> single-file download path
        'https://raw.githubusercontent.com/O/R/main/reference/x.geojson',
        # any non-github host
        'https://example.com/pourpoints/x.geojson',
    ],
)
def test_parse_github_url_returns_none_for_non_folder(url):
    assert parse_github_url(url) is None


def test_flat_names_maps_basenames():
    paths = ['reference/a.geojson', 'reference/b.geojson']
    assert _flat_names(paths) == {
        'reference/a.geojson': 'a.geojson',
        'reference/b.geojson': 'b.geojson',
    }


def test_flat_names_rejects_basename_collision():
    # Two source paths sharing a basename would clobber each other in the flat
    # temp dir; that is an error, not a silently dropped record.
    with pytest.raises(RemoteSourceError, match='Duplicate filename'):
        _flat_names(['reference/x.geojson', 'user/x.geojson'])


def test_materialize_dir_local_path_is_passthrough(tmp_path):
    # A local directory yields unchanged -- no copy, no temp dir.
    with materialize_dir(str(tmp_path)) as local:
        assert local == tmp_path


def test_materialize_file_local_path_is_passthrough(tmp_path):
    src = tmp_path / 'x.geojson'
    src.write_text('{}')
    with materialize_file(str(src)) as local:
        assert local == src


def test_materialize_file_rejects_local_directory(tmp_path):
    # import is single-record; a directory belongs to sync.
    with pytest.raises(IsADirectoryError), materialize_file(str(tmp_path)):
        pass


def test_materialize_file_rejects_github_folder_url():
    url = 'https://github.com/O/R/tree/main/reference'
    with (
        pytest.raises(RemoteSourceError, match='use `pourpoint sync`'),
        materialize_file(url),
    ):
        pass


def test_materialize_dir_rejects_single_file_url():
    url = 'https://raw.githubusercontent.com/O/R/main/reference/x.geojson'
    with (
        pytest.raises(RemoteSourceError, match='needs a local directory'),
        materialize_dir(url),
    ):
        pass
