"""The ingest seam: generic write path, delegation, and the SNODAS ingester.

The generic write path (:meth:`Dataset.write_date_cogs`) commits a whole per-date
directory atomically via the staged-dir seam, so these drive it through the real
path (no monkeypatching of ``write_date_cogs``) on a tiny purpose-built two-variable
spec. The rasters write tiny *real* COGs carrying the ``SOURCE_HASH`` tag (the skip
check reads it back), which is enough to exercise the filename-set completeness,
hash-based skip-if-current, and stale-cleanup logic without a full dataset fixture.
"""

from datetime import date

import pytest

from snowtool.exceptions import IncompleteDatasetDataError, SnowtoolError
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.datasets import SNODAS_SPEC, SnodasIngester
from snowtool.snowdb.datasets.snodas import SNODASInputRasterSet
from snowtool.snowdb.ingest import INGEST_FORMAT_VERSION, IngestResult
from snowtool.snowdb.provenance import versioned_hash
from snowtool.snowdb.spec import DatasetSpec
from snowtool.snowdb.variables import DatasetVariable, Reducer, Unit

from ..conftest import CapturingProgress, make_dataset, write_marker_cog

_MM = Unit(name='mm', scale_factor=1)

# Two distinct versioned source hashes for the write-path tests.
_HASH_A = versioned_hash(INGEST_FORMAT_VERSION, 'a' * 64)
_HASH_B = versioned_hash(INGEST_FORMAT_VERSION, 'b' * 64)


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
    return make_dataset(two, tmp_path / 'db')


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
        write_marker_cog(output_dir / self.out_name, self.source_hash)
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


def _write(dataset, d, rasters, *, source_hash=_HASH_A, force=False, **kwargs):
    """Drive write_date_cogs' (out_names, deferred build) contract from a raster list.

    The write path now takes the declared ``out_names`` and a zero-arg
    ``build_rasters`` (invoked only if the date is not skipped); this adapts a plain
    list of fake rasters to that shape so the test bodies stay about behaviour.
    """
    return dataset.write_date_cogs(
        d,
        frozenset(r.out_name for r in rasters),
        lambda: rasters,
        source_hash=source_hash,
        force=force,
        **kwargs,
    )


def test_write_date_cogs_creates_date_dir_and_writes(two_var_dataset):
    rasters = _full('srcA')
    d = date(2020, 1, 5)

    assert _write(two_var_dataset, d, rasters)

    out = two_var_dataset.date_dir(d)
    assert out.is_dir()
    assert {p.name for p in out.iterdir()} == {'srcA__swe.tif', 'srcA__depth.tif'}
    # Each raster wrote into the staging dir, which was then swapped onto `out`.
    assert all(r.written_to is not None for r in rasters)


def test_write_date_cogs_missing_variable_raises_before_writing(two_var_dataset):
    # A source short a required input variable is caught before any staging.
    with pytest.raises(IncompleteDatasetDataError, match='depth'):
        _write(two_var_dataset, date(2020, 1, 5), [_FakeRaster('srcA', 'swe')])

    assert not two_var_dataset.date_dir(date(2020, 1, 5)).exists()


def test_write_date_cogs_skips_when_current(two_var_dataset):
    d = date(2020, 1, 5)
    _write(two_var_dataset, d, _full('srcA'))
    out = two_var_dataset.date_dir(d)
    before = {p.name: p.stat().st_mtime_ns for p in out.iterdir()}

    # Re-run unchanged (same names, same hash) -> skipped wholesale, returns False.
    rerun = _full('srcA')
    assert _write(two_var_dataset, d, rerun) is False

    assert all(r.written_to is None for r in rerun)
    assert {p.name: p.stat().st_mtime_ns for p in out.iterdir()} == before


def test_write_date_cogs_skip_does_not_build_rasters(two_var_dataset):
    # The build callable is invoked only when a date is (re)built: an already-current
    # date must be skipped without ever calling it (SNODAS never extracts its tar).
    d = date(2020, 1, 5)
    _write(two_var_dataset, d, _full('srcA'))

    built = 0

    def build():
        nonlocal built
        built += 1
        return _full('srcA')

    wrote = two_var_dataset.write_date_cogs(
        d,
        frozenset({'srcA__swe.tif', 'srcA__depth.tif'}),
        build,
        source_hash=_HASH_A,
    )

    assert wrote is False
    assert built == 0


def test_write_date_cogs_same_name_different_bytes_rebuilds(two_var_dataset):
    # A re-release under the *same* filename with different bytes keeps the name
    # set identical, so only the source-hash mismatch catches it.
    d = date(2020, 1, 5)
    _write(two_var_dataset, d, _full('srcA', _HASH_A))
    out = two_var_dataset.date_dir(d)

    rerun = _full('srcA', _HASH_B)
    assert _write(two_var_dataset, d, rerun, source_hash=_HASH_B) is True

    assert all(r.written_to is not None for r in rerun)
    assert two_var_dataset._date_source_hash(out) == _HASH_B


def test_write_date_cogs_missing_hash_tag_rebuilds(two_var_dataset):
    # A legacy date dir (matching names, but its COGs predate SOURCE_HASH) reads as
    # stale and rebuilds rather than wrongly skipping.
    d = date(2020, 1, 5)
    out = two_var_dataset.date_dir(d)
    out.mkdir(parents=True)
    for name in ('srcA__swe.tif', 'srcA__depth.tif'):
        write_marker_cog(out / name, None)

    rerun = _full('srcA')
    assert _write(two_var_dataset, d, rerun) is True

    assert all(r.written_to is not None for r in rerun)
    assert two_var_dataset._date_source_hash(out) == _HASH_A


def test_write_date_cogs_force_rebuilds_current_dir(two_var_dataset):
    d = date(2020, 1, 5)
    _write(two_var_dataset, d, _full('srcA'))

    # force rebuilds even when names + hash both already match.
    rerun = _full('srcA')
    assert _write(two_var_dataset, d, rerun, force=True)

    assert all(r.written_to is not None for r in rerun)


def test_write_date_cogs_reports_progress_per_cog(two_var_dataset):
    progress = CapturingProgress()
    d = date(2020, 1, 5)

    _write(two_var_dataset, d, _full('srcA'), progress=progress)

    # One task for the rebuilt date, advanced once per written COG (swe + depth),
    # labelled with the dataset name and ISO date.
    assert [(t.label, t.total, t.advanced) for t in progress.tasks] == [
        ('test 2020-01-05', 2, 2),
    ]


def test_write_date_cogs_skip_reports_no_progress(two_var_dataset):
    d = date(2020, 1, 5)
    _write(two_var_dataset, d, _full('srcA'))

    # An already-current date returns before the write loop, so no task is opened.
    progress = CapturingProgress()
    assert _write(two_var_dataset, d, _full('srcA'), progress=progress) is False
    assert progress.tasks == []


def test_write_date_cogs_changed_source_replaces_stale(two_var_dataset):
    d = date(2020, 1, 5)
    _write(two_var_dataset, d, _full('srcA'))

    # A differently-named source (e.g. a version bump): same keys, new stems.
    _write(two_var_dataset, d, _full('srcB', _HASH_B), source_hash=_HASH_B)

    out = two_var_dataset.date_dir(d)
    # The wholesale swap dropped srcA's stale COGs by construction.
    assert {p.name for p in out.iterdir()} == {'srcB__swe.tif', 'srcB__depth.tif'}
    swe = two_var_dataset.spec.variables['swe']
    assert two_var_dataset.variable_path(d, swe).name == 'srcB__swe.tif'


def test_write_date_cogs_force_clears_decoy_and_resolves(two_var_dataset):
    d = date(2020, 1, 5)
    out = two_var_dataset.date_dir(d)
    _write(two_var_dataset, d, _full('srcA'))
    # A stale decoy from a prior differently-named source also matches *__swe.tif,
    # which would make swe unresolvable on read.
    (out / 'otherstem__swe.tif').write_text('stale')

    _write(two_var_dataset, d, _full('srcA'), force=True)

    assert {p.name for p in out.iterdir()} == {'srcA__swe.tif', 'srcA__depth.tif'}
    swe = two_var_dataset.spec.variables['swe']
    assert two_var_dataset.variable_path(d, swe).name == 'srcA__swe.tif'


def test_write_date_cogs_post_validate_discards_and_preserves(two_var_dataset):
    d = date(2020, 1, 5)
    out = two_var_dataset.date_dir(d)
    _write(two_var_dataset, d, _full('srcA'))
    good = {p.name for p in out.iterdir()}

    # depth's raster claims depth in out_name (passes pre-check) but writes a
    # misnamed file, so the in-staging post-check fails and the swap is abandoned.
    bad = [
        _FakeRaster('srcB', 'swe'),
        _MisfilingRaster('srcB__depth.tif', 'srcB__WRONG.tif'),
    ]
    with pytest.raises(IncompleteDatasetDataError, match='depth'):
        _write(two_var_dataset, d, bad, source_hash=_HASH_B, force=True)

    # The prior, good date dir is left exactly as it was.
    assert {p.name for p in out.iterdir()} == good


def test_variable_path_duplicate_raises_incomplete(two_var_dataset):
    d = date(2020, 1, 5)
    out = two_var_dataset.date_dir(d)
    _write(two_var_dataset, d, _full('srcA'))
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
                out_names=frozenset({'srcA__swe.tif', 'srcA__depth.tif'}),
                build_rasters=lambda _hash: _full('srcA'),
            )

    recorder = _Recorder()
    ingestable = DatasetSpec(
        name='test',
        grid_params=spec.grid_params,
        variables=(_var('swe'), _var('depth')),
        ingester=recorder,
    )
    ds = make_dataset(ingestable, tmp_path / 'db')
    src = tmp_path / 'src.tar'
    src.write_bytes(b'source bytes')

    result = ds.ingest(src, force=True)

    assert result == IngestResult(ingested=[date(2021, 3, 1)], skipped=[])
    assert recorder.calls == [(src, ds)]


# --- SnodasIngester orchestration ---------------------------------------------
#
# plan() now reads only the tar's *member names* -- no extraction, no GDAL -- so it
# runs over a real archive built from real SNODAS filename stems. Extraction (the
# SNODAS/GDAL boundary) is the one thing stubbed: `extract_archive` is a spy that
# also drops empty header files, and each raster's rasterio-backed `write_cog` is
# swapped for a marker write. So the whole plan -> skip -> build -> write pipeline
# runs on real code, and the skip path is proven to trigger no extraction.


def _snodas_stems(date_str: str = '20190202', hour: str = '05') -> list[str]:
    """A full per-product set of parseable SNODAS filename stems for one date/hour.

    Non-precip products are vcode-agnostic (any 4-char vcode parses), so they reuse
    'lL00'; the two precip variants carry their real 'lL00'/'lL01' split -- yielding
    exactly one stem per :class:`Product`, the complete set the ingester requires.
    """
    stems = []
    for _product, (code, vcode, _unit) in _snodas_products().items():
        vc = vcode or 'lL00'
        stems.append(f'zz_ssmv1{code}S{vc}T0001TTNATS{date_str}{hour}HP001')
    return stems


def _snodas_products():
    from snowtool.snowdb.datasets.snodas import _PRODUCTS

    return _PRODUCTS


def _write_snodas_tar(path, stems: list[str], payload: bytes = b'placeholder') -> None:
    """A real (gzipped-members) SNODAS tar carrying each stem's header + data.

    The member names are what plan() reads; the members' *contents* are irrelevant
    to it (extraction is stubbed), so tiny gzipped placeholders suffice. Header
    members are ``.txt.gz`` (what ``header_stems`` keys on); data rides as ``.dat.gz``.
    A differing ``payload`` yields the *same* member names with different tar bytes --
    the same-name re-release the hash-based skip catches.
    """
    import gzip
    import io
    import tarfile

    def _add(tar, name: str) -> None:
        body = gzip.compress(payload)
        info = tarfile.TarInfo(name)
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))

    with tarfile.open(path, 'w') as tar:
        for stem in stems:
            _add(tar, f'{stem}.txt.gz')
            _add(tar, f'{stem}.dat.gz')


class _ExtractSpy:
    """Records SNODAS extraction calls and drops empty header files per call.

    Stubs the SNODAS/GDAL extraction boundary: it counts invocations (so the skip
    path can assert zero) and writes an empty ``<stem>.txt`` per parsed name into the
    extract dir, so build_rasters' stem->header match resolves without real bytes.
    """

    def __init__(self, stems: list[str]) -> None:
        self.stems = stems
        self.calls = 0

    def __call__(self, source, extract_dir) -> None:
        self.calls += 1
        for stem in self.stems:
            (extract_dir / f'{stem}.txt').write_text('')


def _patch_snodas_boundaries(monkeypatch, spy: _ExtractSpy) -> None:
    """Swap the two SNODAS/GDAL boundaries: extraction (spy) and the rasterio write.

    plan()'s member-name read and build_rasters' stem-matching run for real; only
    the archive extraction and the per-raster COG write (both needing the system
    SNODAS/GDAL stack) are replaced -- the write drops a marker COG carrying the
    (now required, immutable) source_hash the skip check reads back.
    """
    monkeypatch.setattr(
        'snowtool.snowdb.datasets.snodas.SNODASInputRasterSet.extract_archive',
        staticmethod(spy),
    )

    def _marker_write(self, output_dir) -> None:
        write_marker_cog(output_dir / self.out_name, self.source_hash)

    monkeypatch.setattr(
        'snowtool.snowdb.datasets.snodas.SNODASInputRaster.write_cog',
        _marker_write,
    )


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


def test_snodas_set_accepts_pinned_05_timestep():
    # A full product set at the pinned 05 time-step (the daily product) -> no error.
    rs = SNODASInputRasterSet.from_names(_snodas_stems(hour='05'))
    assert rs.date == date(2019, 2, 2)


def test_snodas_set_refuses_other_timestep_hours():
    # Any other time-step hour is refused so a date never mixes revisions.
    with pytest.raises(SnowtoolError, match='pins to the 05 time-step'):
        SNODASInputRasterSet.from_names(_snodas_stems(hour='18'))


def test_snodas_raster_set_refuses_unmasked_region():
    """The unmasked 'us' product is a different lattice; ingest pins to 'zz'."""
    stems = _snodas_stems('20190202')
    stems[0] = stems[0].replace('zz_', 'us_', 1)
    with pytest.raises(SnowtoolError, match='pins to the masked'):
        SNODASInputRasterSet.from_names(stems)


def _snodas_spec(spec):
    """A synthetic-grid snodas dataset spec carrying the real 8 SNODAS variables."""
    return DatasetSpec(
        name='snodas',
        grid_params=spec.grid_params,
        variables=tuple(SNODAS_SPEC.variables.values()),
        ingester=SnodasIngester(),
    )


def test_snodas_ingester_writes_date_cogs(tmp_path, spec, monkeypatch):
    """The SNODAS plan is driven through run_ingest into per-date COGs.

    Only the SNODAS/GDAL extraction + rasterio write are stubbed; plan reads the real
    tar's member names, so the full product set + provenance-named COGs are exercised.
    """
    ds = make_dataset(_snodas_spec(spec), tmp_path / 'db')
    d = date(2019, 2, 2)
    stems = _snodas_stems('20190202')
    tar = tmp_path / 'snodas.tar'
    _write_snodas_tar(tar, stems)
    spy = _ExtractSpy(stems)
    _patch_snodas_boundaries(monkeypatch, spy)

    result = ds.ingest(tar)

    assert result == IngestResult(ingested=[d], skipped=[])
    # One provenance-named COG per product landed.
    assert {p.name for p in ds.date_dir(d).iterdir()} == {f'{s}.tif' for s in stems}
    # Extraction happened exactly once (the one date built).
    assert spy.calls == 1
    # The driver stamped the versioned tar hash on the written COGs.
    assert ds._date_source_hash(ds.date_dir(d)).startswith(
        f'v{INGEST_FORMAT_VERSION}:',
    )


def test_snodas_skip_path_does_no_extraction(tmp_path, spec, monkeypatch):
    """An already-current archive is skipped with ZERO tar extraction.

    The converge-by-default bulk re-run: after the first ingest, re-ingesting the
    identical tar must be skipped on the member-names + hash alone -- build_rasters
    (the only extraction site) is never called, so the spy's call count does not rise.
    """
    ds = make_dataset(_snodas_spec(spec), tmp_path / 'db')
    d = date(2019, 2, 2)
    stems = _snodas_stems('20190202')
    tar = tmp_path / 'snodas.tar'
    _write_snodas_tar(tar, stems)
    spy = _ExtractSpy(stems)
    _patch_snodas_boundaries(monkeypatch, spy)

    assert ds.ingest(tar).ingested == [d]
    assert spy.calls == 1  # the first, real build

    # Re-ingest the identical archive -> skipped, and NO further extraction.
    result = ds.ingest(tar)
    assert result == IngestResult(ingested=[], skipped=[d])
    assert spy.calls == 1  # unchanged: the skip path extracted nothing


def test_snodas_ingester_skips_unchanged_source_and_force_reingests(
    tmp_path,
    spec,
    monkeypatch,
):
    # Converge-by-default: re-ingesting the identical tar is skipped; a same-name
    # tar with different bytes rebuilds; --force always rebuilds.
    ds = make_dataset(_snodas_spec(spec), tmp_path / 'db')
    d = date(2019, 2, 2)
    stems = _snodas_stems('20190202')
    tar = tmp_path / 'snodas.tar'
    _write_snodas_tar(tar, stems)
    spy = _ExtractSpy(stems)
    _patch_snodas_boundaries(monkeypatch, spy)
    a_cog = ds.date_dir(d) / f'{stems[0]}.tif'

    assert ds.ingest(tar).ingested == [d]
    first_mtime = a_cog.stat().st_mtime_ns

    # Same bytes -> skipped, files untouched, no re-extraction.
    result = ds.ingest(tar)
    assert result == IngestResult(ingested=[], skipped=[d])
    assert a_cog.stat().st_mtime_ns == first_mtime
    assert spy.calls == 1

    # Same names, different bytes -> rebuilt (hash mismatch), so a fresh extraction.
    _write_snodas_tar(tar, stems, payload=b're-released under the same name')
    assert ds.ingest(tar).ingested == [d]
    assert spy.calls == 2

    # force rebuilds even when the hash matches.
    result = ds.ingest(tar, force=True)
    assert result.ingested == [d]
    assert spy.calls == 3
