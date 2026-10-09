"""``_get_file``'s write path, driven through an injected fake transport.

The real transport is ``curl_cffi``; these tests pass a ``FakeSession``
through ``_get_file``'s ``session`` seam instead, so the chunk-write,
atomic-rename, and failure-cleanup behavior is exercised with no network.
``test_download_network.py`` covers the real transport.
"""

from datetime import date

import pytest

from curl_cffi.requests.exceptions import HTTPError, RequestException

from snowtool.cli.download import _get_file
from snowtool.snowdb.downloads import SWANNUrl

URL = 'https://example.invalid/data/SNODAS_20260925.tar'
FILENAME = 'SNODAS_20260925.tar'
PART = FILENAME + '.part'


class FakeResponse:
    """A streamed response that yields fixed chunks, or raises."""

    def __init__(self, chunks=(), raises=None):
        self._chunks = chunks
        self._raises = raises
        self.closed = False

    def raise_for_status(self):
        if self._raises is not None:
            raise self._raises

    def iter_content(self, chunk_size):
        yield from self._chunks

    def close(self):
        self.closed = True


class FakeSession:
    """Records how it was called and hands back a prepared response."""

    def __init__(self, response):
        self._response = response
        self.calls = []
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self._response

    def close(self):
        self.closed = True


def test_streams_chunks_to_dest(tmp_path):
    response = FakeResponse(chunks=[b'abc', b'de'])
    session = FakeSession(response)

    result = _get_file(URL, tmp_path, session)

    assert result.status is True
    assert result.detail == ''
    assert (tmp_path / FILENAME).read_bytes() == b'abcde'
    assert not (tmp_path / PART).exists()
    assert response.closed is True


def test_requests_stream_mode(tmp_path):
    """The regression guard: without ``stream=True`` iter_content asserts."""
    session = FakeSession(FakeResponse(chunks=[b'x']))

    _get_file(URL, tmp_path, session)

    url, kwargs = session.calls[0]
    assert url == URL
    assert kwargs['stream'] is True


def test_filename_is_taken_from_the_url(tmp_path):
    session = FakeSession(FakeResponse(chunks=[b'x']))

    _get_file('https://example.invalid/a/b/UA_SWE_v1.nc', tmp_path, session)

    assert (tmp_path / 'UA_SWE_v1.nc').is_file()


def test_creates_missing_destination_directories(tmp_path):
    session = FakeSession(FakeResponse(chunks=[b'x']))
    nested = tmp_path / '2026' / '09'

    result = _get_file(URL, nested, session)

    assert result.status is True
    assert (nested / FILENAME).is_file()


@pytest.mark.parametrize(
    'error',
    [
        HTTPError('HTTP Error 404: '),
        RequestException('connection reset'),
        OSError('No space left on device'),
    ],
)
def test_failure_cleans_up_the_part_file(tmp_path, error):
    response = FakeResponse(raises=error)
    session = FakeSession(response)

    result = _get_file(URL, tmp_path, session)

    assert result.status is False
    assert str(error) in result.detail
    assert not (tmp_path / FILENAME).exists()
    assert not (tmp_path / PART).exists()
    assert response.closed is True


def test_an_injected_session_is_left_open(tmp_path):
    """A caller-supplied session is reused across files, so _get_file keeps it."""
    session = FakeSession(FakeResponse(chunks=[b'x']))

    _get_file(URL, tmp_path, session)

    assert session.closed is False


@pytest.mark.parametrize(
    ('target', 'expected_wy'),
    [
        (date(2026, 9, 30), 2026),  # last day of WY2026
        (date(2026, 10, 1), 2027),  # first day of WY2027
        (date(2026, 10, 5), 2027),
        (date(2027, 1, 15), 2027),
    ],
)
def test_swann_correct_water_year(target, expected_wy):
    url = SWANNUrl._for_date(target).url
    assert f'/WY{expected_wy}/' in url
