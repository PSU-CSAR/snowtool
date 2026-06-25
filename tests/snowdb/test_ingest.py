"""The ingest seam: generic write path, delegation, and the SNODAS ingester."""

from datetime import UTC, date, datetime

import pytest

from snowtool.exceptions import SNODASError
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.datasets import SNODAS_SPEC, SnodasIngester
from snowtool.snowdb.input_rasters import SNODASInputRasterSet
from snowtool.snowdb.spec import DatasetSpec


class _FakeRaster:
    """A WritableRaster that just drops a marker file in the date dir."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.written_to = None

    def write_cog(self, output_dir, force: bool = False) -> None:
        (output_dir / f'{self.name}.tif').write_text('cog')
        self.written_to = output_dir


# --- Dataset.write_date_cogs (the generic write side) ------------------------


def test_write_date_cogs_creates_date_dir_and_writes(dataset):
    rasters = [_FakeRaster('a'), _FakeRaster('b')]

    dataset.write_date_cogs(date(2020, 1, 5), rasters)

    out = dataset.date_dir(date(2020, 1, 5))
    assert out.is_dir()
    assert (out / 'a.tif').is_file()
    assert (out / 'b.tif').is_file()
    assert all(r.written_to == out for r in rasters)


def test_write_date_cogs_existing_dir_without_force_raises(dataset):
    dataset.date_dir(date(2020, 1, 5)).mkdir(parents=True)

    with pytest.raises(FileExistsError, match='already exists'):
        dataset.write_date_cogs(date(2020, 1, 5), [_FakeRaster('a')])


def test_write_date_cogs_force_overwrites(dataset):
    dataset.date_dir(date(2020, 1, 5)).mkdir(parents=True)

    dataset.write_date_cogs(date(2020, 1, 5), [_FakeRaster('a')], force=True)

    assert (dataset.date_dir(date(2020, 1, 5)) / 'a.tif').is_file()


# --- Dataset.ingest (delegation to spec.ingester) ----------------------------


def test_ingest_without_ingester_raises(dataset, tmp_path):
    # The synthetic spec configures no ingester.
    assert dataset.spec.ingester is None

    with pytest.raises(SNODASError, match='no configured ingester'):
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
    with pytest.raises(SNODASError, match='pins to the 05 time-step'):
        SNODASInputRasterSet.validate_revision(
            [_RevisionRaster(5), _RevisionRaster(18)],
        )


def test_snodas_ingester_writes_date_cogs(dataset, tmp_path, monkeypatch):
    """SnodasIngester parses an archive then writes via Dataset.write_date_cogs.

    The archive parsing (which needs the system SNODAS/GDAL stack) is stubbed so
    this exercises only the ingester's orchestration on the synthetic dataset.
    """

    class _FakeSet:
        date = date(2019, 2, 2)

        def __iter__(self):
            yield _FakeRaster('swe')

    def fake_from_archive(source, extract_dir):
        assert source == tmp_path / 'snodas.tar'
        return _FakeSet()

    monkeypatch.setattr(
        'snowtool.snowdb.input_rasters.SNODASInputRasterSet.from_archive',
        staticmethod(fake_from_archive),
    )

    dates = SnodasIngester().ingest(tmp_path / 'snodas.tar', dataset)

    assert dates == [date(2019, 2, 2)]
    assert (dataset.date_dir(date(2019, 2, 2)) / 'swe.tif').is_file()
