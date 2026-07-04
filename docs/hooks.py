"""Build-time hook: emit the HTTP API's OpenAPI spec into the docs build.

Registered under ``hooks:`` in ``mkdocs.yml`` -- a native MkDocs feature, so no
plugin dependency. Adds ``reference/openapi.json`` as a generated file (rendered
by ``reference/http-api.md`` via Redoc).

The single source of truth is ``tests/api/openapi_snapshot.json`` -- the golden
file ``tests/api/test_openapi_snapshot.py`` pins against the served
``/openapi.json``, so it can't silently drift from production. Reusing it means
the docs build needs no live snowdb (``get_app`` would otherwise open a catalog
from config). The snapshot scrubs the git-tag-derived version to a placeholder;
we restore the real ``snowtool.__version__`` so the rendered docs show it.
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import TYPE_CHECKING

from mkdocs.structure.files import File

from snowtool import __version__

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files

_SNAPSHOT = Path('tests/api/openapi_snapshot.json')
_OUTPUT = 'reference/openapi.json'


def on_files(files: Files, config: MkDocsConfig) -> Files:
    spec = json.loads(_SNAPSHOT.read_text())
    spec.setdefault('info', {})['version'] = __version__
    files.append(File.generated(config, _OUTPUT, content=json.dumps(spec, indent=2)))
    return files
