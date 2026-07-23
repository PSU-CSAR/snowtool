"""The Annual NLCD source's download/extract path, driven over a ``file://`` URL.

The real source fetches a ~1.5 GB ``.zip`` from MRLC over HTTPS; that bundle sends
a ``Content-Length`` and no chunked encoding (verified against the live URL), so the
download reads the length header and streams the body in chunks while reporting
byte progress. ``urllib.request.urlopen`` serves ``file://`` URLs through the same
response API (``.headers.get('Content-Length')`` + chunked ``.read``), so pointing
:class:`AnnualNLCD` at a local zip exercises that exact path -- download, byte
progress, extract, cleanup -- with no network and no mock.
"""

import io
import zipfile

from snowtool.snowdb.zones.landcover_source import AnnualNLCD

from ..conftest import CapturingProgress


def _make_bundle(directory, *, stem='bundle', payload=b'GEOTIFF-BYTES' * 500):
    """Write ``<stem>.zip`` holding ``<stem>.tif`` (the member AnnualNLCD extracts).

    Returns ``(zip_path, payload)``; the payload is arbitrary bytes (the test never
    opens it as a raster -- only the download/extract mechanics are under test).
    """
    zip_path = directory / f'{stem}.zip'
    buffer = io.BytesIO()
    # Pin the member's timestamp to the DOS epoch: writestr's default reads the
    # system clock, which zipfile can't encode if it predates 1980.
    member = zipfile.ZipInfo(f'{stem}.tif', date_time=(1980, 1, 1, 0, 0, 0))
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member, payload)
    zip_path.write_bytes(buffer.getvalue())
    return zip_path, payload


def test_ensure_local_downloads_extracts_and_reports_byte_progress(tmp_path):
    src_dir = tmp_path / 'remote'
    src_dir.mkdir()
    zip_path, payload = _make_bundle(src_dir)
    zip_size = zip_path.stat().st_size

    cache = tmp_path / 'cache'
    source = AnnualNLCD(url=zip_path.as_uri(), cache_dir=cache)
    progress = CapturingProgress()

    raster = source._ensure_local(progress=progress)

    # Extracted the member to the cache and dropped the downloaded zip afterward.
    assert raster == cache / 'bundle.tif'
    assert raster.read_bytes() == payload
    assert not (cache / 'bundle.zip').exists()

    # One byte-progress task, its total the zip's Content-Length, advanced fully.
    (task,) = progress.tasks
    assert task.label == 'downloading NLCD'
    assert task.total == zip_size
    assert task.advanced == zip_size


def test_ensure_local_is_cached_after_first_fetch(tmp_path):
    src_dir = tmp_path / 'remote'
    src_dir.mkdir()
    zip_path, _ = _make_bundle(src_dir)

    cache = tmp_path / 'cache'
    source = AnnualNLCD(url=zip_path.as_uri(), cache_dir=cache)
    source._ensure_local()

    # Second call finds the extracted raster and does no download at all (no task).
    progress = CapturingProgress()
    source._ensure_local(progress=progress)

    assert progress.tasks == []


def test_recovers_from_an_interrupted_extract(tmp_path):
    src_dir = tmp_path / 'remote'
    src_dir.mkdir()
    zip_path, payload = _make_bundle(src_dir)

    # A previous run died mid-extract: the zip is cached, a truncated .part
    # sidecar was stranded, and -- the point of extracting through a sidecar --
    # no final raster exists for is_file() to wrongly accept as complete.
    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / 'bundle.zip').write_bytes(zip_path.read_bytes())
    (cache / 'bundle.tif.part').write_bytes(payload[:7])

    source = AnnualNLCD(url=zip_path.as_uri(), cache_dir=cache)
    progress = CapturingProgress()
    raster = source._ensure_local(progress=progress)

    # The cached zip is re-extracted (no re-download: no progress task), the
    # stale sidecar is overwritten and renamed away, and the zip is dropped.
    assert raster.read_bytes() == payload
    assert progress.tasks == []
    assert not (cache / 'bundle.tif.part').exists()
    assert not (cache / 'bundle.zip').exists()
