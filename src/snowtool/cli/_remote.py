"""Materialize a remote pourpoint source to a local directory for import/sync.

CLI transport, not a domain concept: the ``pourpoint import``/``sync`` pipeline on
:class:`~snowtool.snowdb.manager.SnowDbManager` is purely local -- it classifies and
copies ``*.geojson`` files from a path on disk. This module lives in the CLI shell
because it is the thin adapter that lets the ``pourpoint`` commands point those
domain operations at an ``http(s)`` source instead, by fetching the remote file(s)
into a temporary directory that is then handed to the *unchanged* local pipeline.
It constructs no domain objects -- only local paths -- so GitHub tree-URL parsing
and ``GITHUB_TOKEN`` handling stay out of ``snowdb/``.

Two shapes are supported, chosen from the URL:

* A **single file** -- any ``http(s)`` URL not recognized as ``github.com`` (e.g. a
  ``raw.githubusercontent.com`` link): one GET, for ``import`` of one record. Plain
  HTTP has no directory-listing primitive, so this is all a generic URL can offer.
* A **GitHub tree URL** (``https://github.com/<owner>/<repo>/tree/<ref>/<subdir>`` --
  the URL the browser address bar shows when you open a folder): the directory
  listing comes from the GitHub Git Trees API (which, unlike the Contents API, is
  not capped at 1000 entries), and every ``*.geojson`` blob under ``<subdir>`` is
  downloaded from ``raw.githubusercontent.com`` into the temp dir. A bare repo URL
  (no ``/tree/...``) resolves the repo's default branch and lists from its root.
  This is deliberately GitHub-specific -- generic HTTP cannot enumerate a directory.

Everything is stdlib (``urllib`` + ``json``): no HTTP dependency, no ``git`` binary.
``GITHUB_TOKEN`` is used if set (higher rate limits, private repos) but is not
required for a public repo.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from snowtool.exceptions import RemoteSourceError
from snowtool.snowdb.progress import NULL_PROGRESS

if TYPE_CHECKING:
    from collections.abc import Iterator

    from snowtool.snowdb.progress import ProgressReporter

# A folder-view URL: github.com/<owner>/<repo>/tree/<ref>[/<subdir>]. ``ref`` is a
# single path segment (a branch/tag/sha); ``subdir`` is the rest.
_GITHUB_TREE_RE = re.compile(
    r'^https?://github\.com/'
    r'(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?'
    r'/tree/(?P<ref>[^/]+)(?:/(?P<subdir>.+?))?/?$',
)
# A bare repo URL: github.com/<owner>/<repo>. Matched only after the tree pattern,
# so it never swallows a ``/tree/...`` URL.
_GITHUB_REPO_RE = re.compile(
    r'^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$',
)

_API = 'https://api.github.com'
_RAW = 'https://raw.githubusercontent.com'
# Modest fan-out for the per-file raw downloads: enough to hide per-request latency
# over hundreds of small files without hammering the host.
_DOWNLOAD_WORKERS = 8


def is_http_url(src: str) -> bool:
    """True if ``src`` is an ``http(s)`` URL (rather than a local path)."""
    return str(src).startswith(('http://', 'https://'))


@dataclass(frozen=True)
class GitHubTree:
    """GitHub coordinates parsed from a tree (or bare repo) URL.

    ``ref`` is ``None`` when the URL named no branch (a bare repo URL); it is
    resolved to the repo's default branch before any tree read. ``subdir`` is
    ``''`` for the repo root.
    """

    owner: str
    repo: str
    ref: str | None
    subdir: str


def parse_github_url(url: str) -> GitHubTree | None:
    """Parse a ``github.com`` tree or bare-repo URL; ``None`` if it is neither.

    A ``None`` result means "not a GitHub folder" -- the caller treats the URL as a
    single-file download.
    """
    match = _GITHUB_TREE_RE.match(url)
    if match:
        return GitHubTree(
            owner=match['owner'],
            repo=match['repo'],
            ref=match['ref'],
            subdir=(match['subdir'] or '').strip('/'),
        )
    match = _GITHUB_REPO_RE.match(url)
    if match:
        return GitHubTree(
            owner=match['owner'],
            repo=match['repo'],
            ref=None,
            subdir='',
        )
    return None


def _request(url: str, *, accept: str | None = None) -> urllib.request.Request:
    # GitHub requires a User-Agent or it 403s; a token (if present) lifts rate
    # limits and reaches private repos, but a public repo needs neither.
    headers = {'User-Agent': 'snowtool'}
    if accept:
        headers['Accept'] = accept
    token = os.environ.get('GITHUB_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'
    return urllib.request.Request(url, headers=headers)  # noqa: S310


def _get_bytes(url: str, *, accept: str | None = None) -> bytes:
    """GET ``url`` and return the body, mapping any HTTP/URL error to a typed one."""
    try:
        with urllib.request.urlopen(_request(url, accept=accept)) as resp:  # noqa: S310
            return resp.read()
    except urllib.error.HTTPError as e:
        raise RemoteSourceError(f'{url}: HTTP {e.code} {e.reason}') from e
    except urllib.error.URLError as e:
        raise RemoteSourceError(f'{url}: {e.reason}') from e


def _get_json(url: str) -> dict:
    return json.loads(_get_bytes(url, accept='application/vnd.github+json'))


def _resolve_ref(tree: GitHubTree) -> str:
    """Return ``tree.ref``, or the repo's default branch when it is ``None``."""
    if tree.ref is not None:
        return tree.ref
    info = _get_json(f'{_API}/repos/{tree.owner}/{tree.repo}')
    branch = info.get('default_branch')
    if not branch:
        raise RemoteSourceError(
            f'Could not determine the default branch of '
            f'{tree.owner}/{tree.repo}; pass a /tree/<ref>/... URL.',
        )
    return branch


def _list_geojson(tree: GitHubTree) -> list[str]:
    """Repo-relative paths of every ``*.geojson`` blob under ``tree.subdir``.

    Requires a resolved ``tree.ref`` (see :func:`_resolve_ref`). Uses the Git Trees
    API with ``recursive=1`` -- one request that returns the whole subtree, and
    (unlike the Contents API) is not capped at 1000 entries. A truncated response
    means the repo is too large to enumerate this way, which we reject rather than
    silently import a partial set.
    """
    listing = _get_json(
        f'{_API}/repos/{tree.owner}/{tree.repo}/git/trees/{tree.ref}?recursive=1',
    )
    if listing.get('truncated'):
        raise RemoteSourceError(
            f'GitHub tree listing for {tree.owner}/{tree.repo}@{tree.ref} was '
            'truncated (repository too large to enumerate via the trees API).',
        )
    prefix = f'{tree.subdir}/' if tree.subdir else ''
    return sorted(
        entry['path']
        for entry in listing.get('tree', [])
        if entry.get('type') == 'blob'
        and entry['path'].startswith(prefix)
        and entry['path'].endswith('.geojson')
    )


def _flat_names(paths: list[str]) -> dict[str, str]:
    """Map each source path to its basename, rejecting basename collisions.

    The temp dir is flat (the local pipeline globs ``*.geojson`` at its top level),
    so two source files sharing a basename would clobber each other; that is an
    error rather than a silent dropped record.
    """
    names: dict[str, str] = {}
    seen: dict[str, str] = {}
    for path in paths:
        name = Path(path).name
        if name in seen:
            raise RemoteSourceError(
                f'Duplicate filename {name!r} in source ({seen[name]} and {path}); '
                'cannot flatten into one directory.',
            )
        seen[name] = path
        names[path] = name
    return names


def _fetch_github_tree(
    tree: GitHubTree,
    dest_dir: Path,
    progress: ProgressReporter,
) -> Path:
    """Download every ``*.geojson`` under ``tree`` into ``dest_dir`` (flat)."""
    tree = replace(tree, ref=_resolve_ref(tree))
    paths = _list_geojson(tree)
    if not paths:
        where = tree.subdir or '/'
        raise RemoteSourceError(
            f'No .geojson files found under {where!r} in '
            f'{tree.owner}/{tree.repo}@{tree.ref}.',
        )
    names = _flat_names(paths)

    def download(path: str) -> None:
        raw = f'{_RAW}/{tree.owner}/{tree.repo}/{tree.ref}/{path}'
        (dest_dir / names[path]).write_bytes(_get_bytes(raw))

    with (
        progress.track(f'fetching {len(paths)} pourpoint(s)', total=len(paths)) as task,
        ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as pool,
    ):
        futures = [pool.submit(download, path) for path in paths]
        for future in as_completed(futures):
            future.result()
            task.advance()
    return dest_dir


def _fetch_single_file(url: str, dest_dir: Path) -> Path:
    """Download a single ``http(s)`` URL into ``dest_dir`` and return its path."""
    name = Path(urllib.parse.urlparse(url).path).name or 'pourpoint.geojson'
    dest = dest_dir / name
    dest.write_bytes(_get_bytes(url))
    return dest


@contextmanager
def materialize_file(src: str) -> Iterator[Path]:
    """Yield a single local file for ``pourpoint import``.

    A local path is yielded unchanged (a directory is rejected -- ``import`` is
    single-record; use ``sync`` for a folder). A single-file ``http(s)`` URL (e.g. a
    ``raw.githubusercontent.com`` link) is downloaded into a temp dir; a GitHub
    *folder* URL is rejected the same way a local directory is. The temp file (when
    created) is removed on exit, so the caller must consume it inside the ``with``.
    """
    if not is_http_url(src):
        path = Path(src)
        if path.is_dir():
            raise IsADirectoryError(src)
        yield path
        return
    if parse_github_url(src) is not None:
        raise RemoteSourceError(
            f'{src} is a GitHub folder URL; `pourpoint import` takes one record -- '
            'pass a single-file (raw) URL, or use `pourpoint sync` for a folder.',
        )
    with tempfile.TemporaryDirectory(prefix='snowtool-pourpoints-') as tmp:
        yield _fetch_single_file(src, Path(tmp))


@contextmanager
def materialize_dir(
    src: str,
    *,
    progress: ProgressReporter = NULL_PROGRESS,
) -> Iterator[Path]:
    """Yield a local directory of ``*.geojson`` records for ``pourpoint sync``.

    A local path is yielded unchanged (the manager rejects a non-directory). A
    GitHub tree/repo URL yields a temp directory of every ``*.geojson`` under it; a
    non-GitHub (single-file) URL is rejected -- ``sync`` needs a folder. The temp
    directory (when created) is removed on exit, so the caller must consume it
    inside the ``with`` -- which the sync pipeline does, copying records into the
    snowdb before returning.
    """
    if not is_http_url(src):
        yield Path(src)
        return
    tree = parse_github_url(src)
    if tree is None:
        raise RemoteSourceError(
            f'{src} is not a GitHub folder URL; `pourpoint sync` needs a local '
            'directory or a github.com/<owner>/<repo>/tree/<branch>/<subdir> URL.',
        )
    with tempfile.TemporaryDirectory(prefix='snowtool-pourpoints-') as tmp:
        yield _fetch_github_tree(tree, Path(tmp), progress)
