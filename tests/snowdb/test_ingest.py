"""The ingest seam: generic write path, delegation, and the SNODAS ingester.

The generic write path (:meth:`Dataset.write_date_cogs`) commits a whole per-date
directory atomically via the staged-dir seam, so these drive it through the real
path (no monkeypatching of ``write_date_cogs``) on a tiny purpose-built two-variable
spec. The rasters write tiny *real* COGs carrying the ``SOURCE_HASH`` tag (the skip
check reads it back), which is enough to exercise the filename-set completeness,
hash-based skip-if-current, and stale-cleanup logic without a full dataset fixture.
"""

from contextlib import contextmanager
from datetime import UTC, date, datetime

import numpy
import pytest

from rasterio.transform import from_origin

from snowtool.exceptions import IncompleteDatasetDataError, SnowtoolError
from snowtool.snowdb.dataset import INGEST_FORMAT_VERSION, Dataset
from snowtool.snowdb.datasets import SNODAS_SPEC, SnodasIngester
from snowtool.snowdb.datasets.snodas import SNODASInputRasterSet
from snowtool.snowdb.ingest import IngestResult
from snowtool.snowdb.provenance import versioned_hash
from snowtool.snowdb.raster import cog
from snowtool.snowdb.spec import DatasetSpec
from snowtool.snowdb.variables import DatasetVariable, Reducer, Unit

from ..conftest import snodas_swe_name

_MM = Unit(name='mm', scale_factor=1)

# Two distinct versioned source hashes for the write-path tests.
_HASH_A = versioned_hash(INGEST_FORMAT_VERSION, 'a' * 64)
_HASH_B = versioned_hash(INGEST_FORMAT_VERSION, 'b' * 64)


def _write_marker_cog(path, source_hash: str | None) -> None:
    """Write a tiny real COG at ``path``, tagged with ``source_hash`` if given.

    Real (not name-only) so the header-only ``SOURCE_HASH`` read the skip check
    does can open it; ``source_hash=None`` simulates a legacy pre-hash COG.
    """
    tags = {cog.SOURCE_HASH_TAG: source_hash} if source_hash is not None else None
    cog.write_cog(
        path,
        numpy.zeros((16, 16), dtype='int16'),
        transform=from_origin(-100.0, 40.0, 0.01, 0.01),
        tile_size=16,
        predictor=2,
        tags=tags,
    )


def _var(key: str) -> DatasetVariable:
    return DatasetVariable(
        key=key,
        unit=_MM,
        reducer=Reducer.MEAN,
        dtype='int16',
        nodata=-9999.0,
        glob=f'*__{key}.tif',
    )


@pytest.fixture
def two_var_dataset(tmp_path, spec):
    """A minimal two-variable (swe, depth) dataset on the synthetic grid."""
    two = DatasetSpec(
        name='test',
        grid_params=spec.grid_params,
        variables=(_var('swe'), _var('depth')),
    )
    return Dataset.create(two, tmp_path / 'db')


class _FakeRaster:
    """A WritableRaster that drops a tiny real COG into the date dir.

    ``out_name`` is ``<stem>__<key>.tif`` so it resolves against a variable's
    ``*__<key>.tif`` glob; each COG carries ``source_hash`` in its ``SOURCE_HASH``
    tag (what a real ingester stamps, and what the skip check reads back).
    ``written_to`` records whether write_cog actually ran (so a test can assert a
    skip did *not* rewrite).
    """

    def __init__(self, stem: str, key: str, source_hash: str = _HASH_A) -> None:
        self.out_name = f'{stem}__{key}.tif'
        self.source_hash = source_hash
        self.written_to = None

    def write_cog(self, output_dir) -> None:
        _write_marker_cog(output_dir / self.out_name, self.source_hash)
        self.written_to = output_dir


class _MisfilingRaster:
    """A WritableRaster that claims one ``out_name`` but writes a different file.

    Passes the pre-write check (its ``out_name`` covers a variable) yet leaves that
    variable unresolvable on disk -- so only the post-write, in-staging check can
    catch it.
    """

    def __init__(self, out_name: str, actual_name: str) -> None:
        self.out_name = out_name
        self._actual = actual_name

    def write_cog(self, output_dir) -> None:
        (output_dir / self._actual).write_text('cog')


# --- Dataset.write_date_cogs (the generic write side) ------------------------


def _full(stem: str, source_hash: str = _HASH_A) -> list[_FakeRaster]:
    return [
        _FakeRaster(stem, 'swe', source_hash),
        _FakeRaster(stem, 'depth', source_hash),
    ]


def test_write_date_cogs_creates_date_dir_and_writes(two_var_dataset):
    rasters = _full('srcA')
    d = date(2020, 1, 5)

    assert two_var_dataset.write_date_cogs(d, rasters, source_hash=_HASH_A)

    out = two_var_dataset.date_dir(d)
    assert out.is_dir()
    assert {p.name for p in out.iterdir()} == {'srcA__swe.tif', 'srcA__depth.tif'}
    # Each raster wrote into the staging dir, which was then swapped onto `out`.
    assert all(r.written_to is not None for r in rasters)


def test_write_date_cogs_missing_variable_raises_before_writing(two_var_dataset):
    # A source short a required input variable is caught before any staging.
    with pytest.raises(IncompleteDatasetDataError, match='depth'):
        two_var_dataset.write_date_cogs(
            date(2020, 1, 5),
            [_FakeRaster('srcA', 'swe')],
            source_hash=_HASH_A,
        )

    assert not two_var_dataset.date_dir(date(2020, 1, 5)).exists()


def test_write_date_cogs_skips_when_current(two_var_dataset):
    d = date(2020, 1, 5)
    two_var_dataset.write_date_cogs(d, _full('srcA'), source_hash=_HASH_A)
    out = two_var_dataset.date_dir(d)
    before = {p.name: p.stat().st_mtime_ns for p in out.iterdir()}

    # Re-run unchanged (same names, same hash) -> skipped wholesale, returns False.
    rerun = _full('srcA')
    assert two_var_dataset.write_date_cogs(d, rerun, source_hash=_HASH_A) is False

    assert all(r.written_to is None for r in rerun)
    assert {p.name: p.stat().st_mtime_ns for p in out.iterdir()} == before


def test_write_date_cogs_same_name_different_bytes_rebuilds(two_var_dataset):
    # A re-release under the *same* filename with different bytes keeps the name
    # set identical, so only the source-hash mismatch catches it.
    d = date(2020, 1, 5)
    two_var_dataset.write_date_cogs(d, _full('srcA', _HASH_A), source_hash=_HASH_A)
    out = two_var_dataset.date_dir(d)

    rerun = _full('srcA', _HASH_B)
    assert two_var_dataset.write_date_cogs(d, rerun, source_hash=_HASH_B) is True

    assert all(r.written_to is not None for r in rerun)
    assert two_var_dataset._date_source_hash(out) == _HASH_B


def test_write_date_cogs_missing_hash_tag_rebuilds(two_var_dataset):
    # A legacy date dir (matching names, but its COGs predate SOURCE_HASH) reads as
    # stale and rebuilds rather than wrongly skipping.
    d = date(2020, 1, 5)
    out = two_var_dataset.date_dir(d)
    out.mkdir(parents=True)
    for name in ('srcA__swe.tif', 'srcA__depth.tif'):
        _write_marker_cog(out / name, None)

    rerun = _full('srcA')
    assert two_var_dataset.write_date_cogs(d, rerun, source_hash=_HASH_A) is True

    assert all(r.written_to is not None for r in rerun)
    assert two_var_dataset._date_source_hash(out) == _HASH_A


def test_write_date_cogs_force_rebuilds_current_dir(two_var_dataset):
    d = date(2020, 1, 5)
    two_var_dataset.write_date_cogs(d, _full('srcA'), source_hash=_HASH_A)

    # force rebuilds even when names + hash both already match.
    rerun = _full('srcA')
    assert two_var_dataset.write_date_cogs(d, rerun, source_hash=_HASH_A, force=True)

    assert all(r.written_to is not None for r in rerun)


class _RecordingProgress:
    """A ProgressReporter that records each tracked task's (label, total) and advances.

    Injected through the ``progress`` seam (no mocks) so a test can assert the write
    loop opens one task per rebuilt date and advances it once per COG.
    """

    def __init__(self) -> None:
        # One (label, total, advances) triple per track() context opened.
        self.tasks: list[tuple[str, int | None, int]] = []

    @contextmanager
    def track(self, label, *, total=None):
        entry = [label, total, 0]

        class _Task:
            def advance(self, n: int = 1) -> None:
                entry[2] += n

        self.tasks.append(entry)
        yield _Task()


def test_write_date_cogs_reports_progress_per_cog(two_var_dataset):
    progress = _RecordingProgress()
    d = date(2020, 1, 5)

    two_var_dataset.write_date_cogs(
        d,
        _full('srcA'),
        source_hash=_HASH_A,
        progress=progress,
    )

    # One task for the rebuilt date, advanced once per written COG (swe + depth),
    # labelled with the dataset name and ISO date.
    assert progress.tasks == [['test 2020-01-05', 2, 2]]


def test_write_date_cogs_skip_reports_no_progress(two_var_dataset):
    d = date(2020, 1, 5)
    two_var_dataset.write_date_cogs(d, _full('srcA'), source_hash=_HASH_A)

    # An already-current date returns before the write loop, so no task is opened.
    progress = _RecordingProgress()
    assert (
        two_var_dataset.write_date_cogs(
            d,
            _full('srcA'),
            source_hash=_HASH_A,
            progress=progress,
        )
        is False
    )
    assert progress.tasks == []


def test_write_date_cogs_changed_source_replaces_stale(two_var_dataset):
    d = date(2020, 1, 5)
    two_var_dataset.write_date_cogs(d, _full('srcA'), source_hash=_HASH_A)

    # A differently-named source (e.g. a version bump): same keys, new stems.
    two_var_dataset.write_date_cogs(d, _full('srcB', _HASH_B), source_hash=_HASH_B)

    out = two_var_dataset.date_dir(d)
    # The wholesale swap dropped srcA's stale COGs by construction.
    assert {p.name for p in out.iterdir()} == {'srcB__swe.tif', 'srcB__depth.tif'}
    swe = two_var_dataset.spec.variables['swe']
    assert two_var_dataset.variable_path(d, swe).name == 'srcB__swe.tif'


def test_write_date_cogs_force_clears_decoy_and_resolves(two_var_dataset):
    d = date(2020, 1, 5)
    out = two_var_dataset.date_dir(d)
    two_var_dataset.write_date_cogs(d, _full('srcA'), source_hash=_HASH_A)
    # A stale decoy from a prior differently-named source also matches *__swe.tif,
    # which would make swe unresolvable on read.
    (out / 'otherstem__swe.tif').write_text('stale')

    two_var_dataset.write_date_cogs(d, _full('srcA'), source_hash=_HASH_A, force=True)

    assert {p.name for p in out.iterdir()} == {'srcA__swe.tif', 'srcA__depth.tif'}
    swe = two_var_dataset.spec.variables['swe']
    assert two_var_dataset.variable_path(d, swe).name == 'srcA__swe.tif'


def test_write_date_cogs_post_validate_discards_and_preserves(two_var_dataset):
    d = date(2020, 1, 5)
    out = two_var_dataset.date_dir(d)
    two_var_dataset.write_date_cogs(d, _full('srcA'), source_hash=_HASH_A)
    good = {p.name for p in out.iterdir()}

    # depth's raster claims depth in out_name (passes pre-check) but writes a
    # misnamed file, so the in-staging post-check fails and the swap is abandoned.
    bad = [
        _FakeRaster('srcB', 'swe'),
        _MisfilingRaster('srcB__depth.tif', 'srcB__WRONG.tif'),
    ]
    with pytest.raises(IncompleteDatasetDataError, match='depth'):
        two_var_dataset.write_date_cogs(d, bad, source_hash=_HASH_B, force=True)

    # The prior, good date dir is left exactly as it was.
    assert {p.name for p in out.iterdir()} == good


def test_variable_path_duplicate_raises_incomplete(two_var_dataset):
    d = date(2020, 1, 5)
    out = two_var_dataset.date_dir(d)
    two_var_dataset.write_date_cogs(d, _full('srcA'), source_hash=_HASH_A)
    # A stale duplicate that slipped in out-of-band makes swe ambiguous.
    (out / 'otherstem__swe.tif').write_text('stale')

    swe = two_var_dataset.spec.variables['swe']
    with pytest.raises(IncompleteDatasetDataError, match='swe'):
        two_var_dataset.variable_path(d, swe)


def test_raster_collection_flags_incomplete_date(tmp_path):
    from snowtool.snowdb.raster.collection import RasterCollection
    from snowtool.snowdb.raster.tiled import DataRaster

    def touch(name):
        p = tmp_path / name
        p.touch()
        return p

    swe, depth = _var('swe'), _var('depth')
    d1, d2 = date(2020, 1, 1), date(2020, 1, 2)
    rasters = {
        swe: [DataRaster(touch('a__swe.tif'), d1), DataRaster(touch('b__swe.tif'), d2)],
        # d2 is missing its depth COG -> a partial date on disk.
        depth: [DataRaster(touch('a__depth.tif'), d1)],
    }

    with pytest.raises(IncompleteDatasetDataError, match='depth'):
        RasterCollection(rasters=rasters, dataset_name='test')


# --- Dataset.ingest (delegation to spec.ingester) ----------------------------


def test_ingest_without_ingester_raises(dataset, tmp_path):
    # The synthetic spec configures no ingester.
    assert dataset.spec.ingester is None

    with pytest.raises(SnowtoolError, match='no configured ingester'):
        dataset.ingest(tmp_path / 'archive.tar')


def test_ingest_delegates_to_ingester(tmp_path, spec):
    from snowtool.snowdb.ingest import DateIngest

    class _Recorder:
        def __init__(self):
            self.calls = []

        def plan(self, source, ds):
            self.calls.append((source, ds))
            yield DateIngest(
                date=date(2021, 3, 1),
                source_files=[source],
                build_rasters=lambda _hash: _full('srcA'),
            )

    recorder = _Recorder()
    ingestable = DatasetSpec(
        name='test',
        grid_params=spec.grid_params,
        variables=(_var('swe'), _var('depth')),
        ingester=recorder,
    )
    ds = Dataset.create(ingestable, tmp_path / 'db')
    src = tmp_path / 'src.tar'
    src.write_bytes(b'source bytes')

    result = ds.ingest(src, force=True)

    assert result == IngestResult(ingested=[date(2021, 3, 1)], skipped=[])
    assert recorder.calls == [(src, ds)]


# --- SnodasIngester orchestration (no GDAL: archive parsing is faked) ---------


def test_snodas_spec_has_a_snodas_ingester():
    assert isinstance(SNODAS_SPEC.ingester, SnodasIngester)


def test_snodas_ingest_rejects_a_directory_source(tmp_path):
    # One archive == one date: a directory earns a precise typed error, not
    # tarfile's raw IsADirectoryError.
    ds = Dataset(SNODAS_SPEC, tmp_path / 'db')
    source = tmp_path / 'archives'
    source.mkdir()
    with pytest.raises(SnowtoolError, match='got a directory'):
        ds.ingest(source)


class _RevisionRaster:
    """Minimal stand-in carrying just the time-step datetime the pin inspects."""

    def __init__(self, hour: int) -> None:
        self.datetime = datetime(2020, 1, 5, hour, tzinfo=UTC)


def test_snodas_set_accepts_pinned_05_timestep():
    # All rasters at the pinned 05 time-step (the daily product) -> no error.
    SNODASInputRasterSet.validate_revision([_RevisionRaster(5), _RevisionRaster(5)])


def test_snodas_set_refuses_other_timestep_hours():
    # Any other time-step hour is refused so a date never mixes revisions.
    with pytest.raises(SnowtoolError, match='pins to the 05 time-step'):
        SNODASInputRasterSet.validate_revision(
            [_RevisionRaster(5), _RevisionRaster(18)],
        )


def _patch_snodas_archive(monkeypatch, swe_name, ingest_date):
    """Stub the SNODAS extract/parse seam with a fresh real-COG raster.

    The archive parsing (which needs the system SNODAS/GDAL stack) is stubbed: the
    extraction is a no-op and ``from_extracted`` returns a one-raster set parsed
    once, whose ``build_rasters`` stamps the driver-computed source hash onto its
    single raster before writing. So the driver's hashing, per-date skip, and write
    orchestration run over real code.
    """

    class _FakeRasterFile:
        def __init__(self) -> None:
            self.out_name = f'{swe_name}.tif'
            self.source_hash: str | None = None

        def write_cog(self, output_dir) -> None:
            _write_marker_cog(output_dir / self.out_name, self.source_hash)

    class _FakeSet:
        def __init__(self) -> None:
            self.date = ingest_date
            self._raster = _FakeRasterFile()

        def __iter__(self):
            yield self._raster

        def build_rasters(self, source_hash: str):
            self._raster.source_hash = source_hash
            return [self._raster]

    monkeypatch.setattr(
        'snowtool.snowdb.datasets.snodas.SNODASInputRasterSet.extract_archive',
        staticmethod(lambda source, extract_dir: None),
    )
    monkeypatch.setattr(
        'snowtool.snowdb.datasets.snodas.SNODASInputRasterSet.from_extracted',
        classmethod(lambda cls, extract_dir: _FakeSet()),
    )


def test_snodas_ingester_writes_date_cogs(tmp_path, spec, monkeypatch):
    """The SNODAS plan is driven through run_ingest into per-date COGs.

    The archive parsing (which needs the system SNODAS/GDAL stack) is stubbed so
    this exercises only the ingest orchestration on a single-variable synthetic
    snodas dataset (so the fake set of one COG still satisfies date completeness).
    """
    mini = DatasetSpec(
        name='snodas',
        grid_params=spec.grid_params,
        variables=(SNODAS_SPEC.variables['swe'],),
        ingester=SnodasIngester(),
    )
    ds = Dataset.create(mini, tmp_path / 'db')
    swe_name = snodas_swe_name('20190202')
    d = date(2019, 2, 2)
    # The driver hashes the source tar, so it must exist on disk.
    tar = tmp_path / 'snodas.tar'
    tar.write_bytes(b'fake snodas archive bytes')
    _patch_snodas_archive(monkeypatch, swe_name, d)

    result = ds.ingest(tar)

    assert result == IngestResult(ingested=[d], skipped=[])
    cog_path = ds.date_dir(d) / f'{swe_name}.tif'
    assert cog_path.is_file()
    # The driver stamped the versioned tar hash on the written COG.
    assert ds._date_source_hash(ds.date_dir(d)).startswith(
        f'v{INGEST_FORMAT_VERSION}:',
    )


def _mini_snodas(tmp_path, spec):
    return Dataset.create(
        DatasetSpec(
            name='snodas',
            grid_params=spec.grid_params,
            variables=(SNODAS_SPEC.variables['swe'],),
            ingester=SnodasIngester(),
        ),
        tmp_path / 'db',
    )


def test_snodas_ingester_skips_unchanged_source_and_force_reingests(
    tmp_path,
    spec,
    monkeypatch,
):
    # Converge-by-default: re-ingesting the identical tar is skipped; a same-name
    # tar with different bytes rebuilds; --force always rebuilds.
    d = date(2019, 2, 2)
    swe_name = snodas_swe_name('20190202')
    ds = _mini_snodas(tmp_path, spec)
    _patch_snodas_archive(monkeypatch, swe_name, d)
    tar = tmp_path / 'snodas.tar'
    tar.write_bytes(b'archive v1')
    cog_path = ds.date_dir(d) / f'{swe_name}.tif'

    assert ds.ingest(tar).ingested == [d]
    first_mtime = cog_path.stat().st_mtime_ns

    # Same bytes -> skipped, file untouched.
    result = ds.ingest(tar)
    assert result == IngestResult(ingested=[], skipped=[d])
    assert cog_path.stat().st_mtime_ns == first_mtime

    # Same name, different bytes -> rebuilt (hash mismatch).
    tar.write_bytes(b'archive v2 -- re-released under the same name')
    assert ds.ingest(tar).ingested == [d]

    # force rebuilds even when the hash matches.
    result = ds.ingest(tar, force=True)
    assert result.ingested == [d]
