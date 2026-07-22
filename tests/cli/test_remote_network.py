"""Real GitHub reads against the authoritative BAGIS-pourpoints repo.

Deselected by default (``network`` marker); run with ``pytest -m network``. These
pin that the trees-API enumeration and raw single-file download work against the
live public repo -- they do *not* download the whole tree (that is what the CLI
sync does), only list it and fetch one record.
"""

import json

from dataclasses import replace

import pytest

from snowtool.cli._remote import (
    GitHubTree,
    _fetch_single_file,
    _list_geojson,
    _resolve_ref,
)

pytestmark = pytest.mark.network

_OWNER = 'PSU-CSAR'
_REPO = 'BAGIS-pourpoints'
_SUBDIR = 'reference'


def _reference_tree() -> GitHubTree:
    """The ``reference/`` folder pinned to the repo's current default branch."""
    ref = _resolve_ref(GitHubTree(_OWNER, _REPO, None, ''))
    return GitHubTree(_OWNER, _REPO, ref, _SUBDIR)


def test_resolve_ref_finds_default_branch():
    assert _resolve_ref(GitHubTree(_OWNER, _REPO, None, ''))


def test_list_geojson_enumerates_reference_dir():
    tree = _reference_tree()
    paths = _list_geojson(tree)

    # The reference dir holds >1000 records -- past the Contents API's cap, so this
    # proves the trees-API path is what makes enumeration complete.
    assert len(paths) > 1000
    assert all(p.startswith(f'{_SUBDIR}/') and p.endswith('.geojson') for p in paths)


def test_fetch_single_file_downloads_a_record(tmp_path):
    tree = _reference_tree()
    first = _list_geojson(tree)[0]
    url = f'https://raw.githubusercontent.com/{_OWNER}/{_REPO}/{tree.ref}/{first}'

    dest = _fetch_single_file(url, tmp_path)

    assert dest.parent == tmp_path
    # A real, parseable pourpoint record came back.
    record = json.loads(dest.read_text())
    assert record['type'] in {'Feature', 'GeometryCollection'}
    assert record['id']


def test_list_geojson_empty_subdir_is_reported():
    tree = _reference_tree()
    empty = replace(tree, subdir='definitely/not/a/real/subdir')
    assert _list_geojson(empty) == []
