"""Golden-file test pinning the OpenAPI contract as a whole.

``test_openapi.py`` asserts the document *builds* for the built-in datasets;
this pins its exact *shape*. The per-dataset generated response models (each
spec's ``zonal_stat(s)_model``), the ``ZoneRef`` discriminated union, and the
pourpoint ``Feature`` models are the client contract -- nothing else in the suite
would catch a rename of e.g. ``mean_swe_mm`` or a reshaped union, since the
behavior tests only check the fields they use. ``test_client`` already builds the
app over the real ``DATASET_TEMPLATES`` (see ``tests/api/conftest.py``), so the
snapshot is the production contract, not a synthetic one.
"""

from __future__ import annotations

import json
import os

from pathlib import Path
from typing import Any

import pytest

from snowtool import __version__ as snowtool_version

SNAPSHOT_PATH = Path(__file__).parent / 'openapi_snapshot.json'
VERSION_PLACEHOLDER = '0'
UPDATE_ENV_VAR = 'SNOWTOOL_UPDATE_SNAPSHOTS'


def _scrub_version(value: Any) -> Any:
    """Replace any embedded ``snowtool.__version__`` string with a placeholder.

    ``__version__`` is git-tag-derived (hatch-vcs) and changes on every commit, so
    it can't appear verbatim in a committed snapshot. It shows up in the OpenAPI
    document as the ``VersionInfo.version`` field's JSON-schema ``default``
    (``snowtool.api.models.root.VersionInfo`` defaults that field to
    ``__version__``); walking the whole document (rather than poking one known
    path) also catches ``info.version`` if a future change ever wires that to the
    package version too.
    """
    if isinstance(value, dict):
        return {key: _scrub_version(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub_version(item) for item in value]
    if value == snowtool_version:
        return VERSION_PLACEHOLDER
    return value


def _normalize(schema: dict[str, Any]) -> str:
    scrubbed = _scrub_version(schema)
    # Belt-and-suspenders: pin info.version explicitly too. Today FastAPI defaults
    # it to a hardcoded '0.1.0' (get_app never passes version=), but that's an
    # implementation detail worth not depending on here.
    scrubbed['info']['version'] = VERSION_PLACEHOLDER
    return json.dumps(scrubbed, indent=2, sort_keys=True) + '\n'


def test_openapi_schema_matches_golden_snapshot(test_client) -> None:
    """The served ``/openapi.json``, normalized, must match the committed golden file.

    On mismatch or first run, set ``SNOWTOOL_UPDATE_SNAPSHOTS=1`` to (re)write the
    file, then re-run without it to confirm -- the write always fails the test so a
    contract change can never slip by silently. A diff here is a client-facing API
    contract change; review it before committing the updated snapshot.
    """
    schema = test_client.get('/openapi.json').json()
    normalized = _normalize(schema)

    if os.environ.get(UPDATE_ENV_VAR):
        SNAPSHOT_PATH.write_text(normalized)
        pytest.fail('snapshot updated -- review the diff and re-run')

    if not SNAPSHOT_PATH.exists():
        SNAPSHOT_PATH.write_text(normalized)
        pytest.fail(
            f'no golden snapshot at {SNAPSHOT_PATH} -- wrote it now; '
            'review the diff and re-run',
        )

    golden = SNAPSHOT_PATH.read_text()
    assert normalized == golden, (
        'OpenAPI schema drifted from tests/api/openapi_snapshot.json -- this is a '
        'client-facing API contract change. If intentional, regenerate with '
        f'{UPDATE_ENV_VAR}=1 uv run pytest tests/api/test_openapi_snapshot.py, '
        'review the diff, and commit the updated snapshot.'
    )
