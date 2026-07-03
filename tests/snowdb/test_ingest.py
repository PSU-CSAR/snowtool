"""The ingest seam: generic write path, delegation, and the SNODAS ingester.

The generic write path (:meth:`Dataset.write_date_cogs`) commits a whole per-date
directory atomically via the staged-dir seam, so these drive it through the real
path (no monkeypatching of ``write_date_cogs``) on a tiny purpose-built two-variable
spec: name-only marker files are enough to exercise the filename-set completeness,
skip-if-current, and stale-cleanup logic without decoding COGs.
"""

from datetime import UTC, date, datetime

import pytest

from snowtool.exceptions import IncompleteDatasetDataError, SnowtoolError
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.datasets import SNODAS_SPEC, SnodasIngester
from snowtool.snowdb.datasets.snodas import SNODASInputRasterSet
from snowtool.snowdb.spec import DatasetSpec
from snowtool.snowdb.variables import DatasetVariable, Reducer, Unit

from ..conftest import snodas_swe_name

_MM = Unit(name='mm', scale_factor=1)


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
    """A WritableRaster that drops a name-only marker COG into the date dir.

    ``out_name`` is ``<stem>__<key>.tif`` so it resolves against a variable's
    ``*__<key>.tif`` glob; ``written_to`` records whether write_cog actually ran
    (so a test can assert a skip did *not* rewrite).
    """

    def __init__(self, stem: str, key: str) -> None:
        self.out_name = f'{stem}__{key}.tif'
        self.written_to = None

    def write_cog(self, output_dir, force: bool = False) -> None:
        (output_dir / self.out_name).write_text('cog')
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

    def write_cog(self, output_dir, force: bool = False) -> None:
        (output_dir / self._actual).write_text('cog')


# --- Dataset.write_date_cogs (the generic write side) ------------------------


def _full(stem: str) -> list[_FakeRaster]:
    return [_FakeRaster(stem, 'swe'), _FakeRaster(stem, 'depth')]


def test_write_date_cogs_creates_date_dir_and_writes(two_var_dataset):
    rasters = _full('srcA')

    two_var_dataset.write_date_cogs(date(2020, 1, 5), rasters)

    out = two_var_dataset.date_dir(date(2020, 1, 5))
    assert out.is_dir()
    assert {p.name for p in out.iterdir()} == {'srcA__swe.tif', 'srcA__depth.tif'}
    # Each raster wrote into the staging dir, which was then swapped onto `out`.
    assert all(r.written_to is not None for r in rasters)


def test_write_date_cogs_missing_variable_raises_before_writing(two_var_dataset):
    # A source short a required input variable is caught before any staging.
    with pytest.raises(IncompleteDatasetDataError, match='depth'):
        two_var_dataset.write_date_cogs(date(2020, 1, 5), [_FakeRaster('srcA', 'swe')])

    assert not two_var_dataset.date_dir(date(2020, 1, 5)).exists()


def test_write_date_cogs_skips_when_current(two_var_dataset):
    d = date(2020, 1, 5)
    two_var_dataset.write_date_cogs(d, _full('srcA'))
    out = two_var_dataset.date_dir(d)
    before = {p.name: p.stat().st_mtime_ns for p in out.iterdir()}

    # Re-run unchanged -> the complete, matching date dir is skipped wholesale.
    rerun = _full('srcA')
    two_var_dataset.write_date_cogs(d, rerun)

    assert all(r.written_to is None for r in rerun)
    assert {p.name: p.stat().st_mtime_ns for p in out.iterdir()} == before


def test_write_date_cogs_force_rebuilds_current_dir(two_var_dataset):
    d = date(2020, 1, 5)
    two_var_dataset.write_date_cogs(d, _full('srcA'))

    rerun = _full('srcA')
    two_var_dataset.write_date_cogs(d, rerun, force=True)

    assert all(r.written_to is not None for r in rerun)


def test_write_date_cogs_changed_source_replaces_stale(two_var_dataset):
    d = date(2020, 1, 5)
    two_var_dataset.write_date_cogs(d, _full('srcA'))

    # A differently-named source (e.g. a version bump): same keys, new stems.
    two_var_dataset.write_date_cogs(d, _full('srcB'))

    out = two_var_dataset.date_dir(d)
    # The wholesale swap dropped srcA's stale COGs by construction.
    assert {p.name for p in out.iterdir()} == {'srcB__swe.tif', 'srcB__depth.tif'}
    swe = two_var_dataset.spec.variables['swe']
    assert two_var_dataset.variable_path(d, swe).name == 'srcB__swe.tif'


def test_write_date_cogs_force_clears_decoy_and_resolves(two_var_dataset):
    d = date(2020, 1, 5)
    out = two_var_dataset.date_dir(d)
    two_var_dataset.write_date_cogs(d, _full('srcA'))
    # A stale decoy from a prior differently-named source also matches *__swe.tif,
    # which would make swe unresolvable on read.
    (out / 'otherstem__swe.tif').write_text('stale')

    two_var_dataset.write_date_cogs(d, _full('srcA'), force=True)

    assert {p.name for p in out.iterdir()} == {'srcA__swe.tif', 'srcA__depth.tif'}
    swe = two_var_dataset.spec.variables['swe']
    assert two_var_dataset.variable_path(d, swe).name == 'srcA__swe.tif'


def test_write_date_cogs_post_validate_discards_and_preserves(two_var_dataset):
    d = date(2020, 1, 5)
    out = two_var_dataset.date_dir(d)
    two_var_dataset.write_date_cogs(d, _full('srcA'))
    good = {p.name for p in out.iterdir()}

    # depth's raster claims depth in out_name (passes pre-check) but writes a
    # misnamed file, so the in-staging post-check fails and the swap is abandoned.
    bad = [
        _FakeRaster('srcB', 'swe'),
        _MisfilingRaster('srcB__depth.tif', 'srcB__WRONG.tif'),
    ]
    with pytest.raises(IncompleteDatasetDataError, match='depth'):
        two_var_dataset.write_date_cogs(d, bad, force=True)

    # The prior, good date dir is left exactly as it was.
    assert {p.name for p in out.iterdir()} == good


def test_variable_path_duplicate_raises_incomplete(two_var_dataset):
    d = date(2020, 1, 5)
    out = two_var_dataset.date_dir(d)
    two_var_dataset.write_date_cogs(d, _full('srcA'))
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
        RasterCollection(query=object(), rasters=rasters, dataset_name='test')


# --- Dataset.ingest (delegation to spec.ingester) ----------------------------


def test_ingest_without_ingester_raises(dataset, tmp_path):
    # The synthetic spec configures no ingester.
    assert dataset.spec.ingester is None

    with pytest.raises(SnowtoolError, match='no configured ingester'):
        dataset.ingest(tmp_path / 'archive.tar')


def test_ingest_delegates_to_ingester(tmp_path, spec, source_dem):
    class _Recorder:
        def __init__(self):
            self.calls = []

        def ingest(self, source, ds, *, force=False):
            self.calls.append((source, ds, force))
            return [date(2021, 3, 1)]

    recorder = _Recorder()
    ingestable = DatasetSpec(
        name='test',
        grid_params=spec.grid_params,
        ingester=recorder,
    )
    ds = Dataset.create(ingestable, tmp_path / 'db', source_dem)

    result = ds.ingest(tmp_path / 'src.tar', force=True)

    assert result == [date(2021, 3, 1)]
    assert recorder.calls == [(tmp_path / 'src.tar', ds, True)]


# --- SnodasIngester orchestration (no GDAL: archive parsing is faked) ---------


def test_snodas_spec_has_a_snodas_ingester():
    assert isinstance(SNODAS_SPEC.ingester, SnodasIngester)


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


def test_snodas_ingester_writes_date_cogs(tmp_path, spec, monkeypatch):
    """SnodasIngester parses an archive then writes via Dataset.write_date_cogs.

    The archive parsing (which needs the system SNODAS/GDAL stack) is stubbed so
    this exercises only the ingester's orchestration on a single-variable synthetic
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

    class _FakeRasterFile:
        out_name = f'{swe_name}.tif'

        def write_cog(self, output_dir, force: bool = False) -> None:
            (output_dir / self.out_name).write_text('cog')

    class _FakeSet:
        date = date(2019, 2, 2)

        def __iter__(self):
            yield _FakeRasterFile()

    def fake_from_archive(source, extract_dir):
        assert source == tmp_path / 'snodas.tar'
        return _FakeSet()

    monkeypatch.setattr(
        'snowtool.snowdb.datasets.snodas.SNODASInputRasterSet.from_archive',
        staticmethod(fake_from_archive),
    )

    dates = SnodasIngester().ingest(tmp_path / 'snodas.tar', ds)

    assert dates == [date(2019, 2, 2)]
    assert (ds.date_dir(date(2019, 2, 2)) / f'{swe_name}.tif').is_file()
