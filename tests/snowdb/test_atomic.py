"""atomic_write_text / atomic_copy / staged_dir: crash-consistent persistence.

These are pure filesystem primitives with no snowdb domain shape, so they are
exercised directly with tmp_path rather than the synthetic-grid fixtures --
there is nothing here for the grid/spec machinery to add.
"""

from __future__ import annotations

import pytest

from snowtool.snowdb.atomic import atomic_copy, atomic_write_text, staged_dir


def _write_then_blow_up(staging):
    """Write a partial member into ``staging``, then fail -- for raises-blocks."""
    (staging / 'partial.tif').write_text('partial')
    raise RuntimeError('boom')


class TestAtomicWriteText:
    def test_creates_file(self, tmp_path):
        path = tmp_path / 'out.txt'
        atomic_write_text(path, 'hello\n')
        assert path.read_text() == 'hello\n'

    def test_replaces_existing_content(self, tmp_path):
        path = tmp_path / 'out.txt'
        path.write_text('old\n')
        atomic_write_text(path, 'new\n')
        assert path.read_text() == 'new\n'

    def test_leaves_no_temp_debris(self, tmp_path):
        path = tmp_path / 'out.txt'
        atomic_write_text(path, 'hello\n')
        assert [p.name for p in tmp_path.iterdir()] == ['out.txt']

    def test_failure_leaves_dest_unchanged_and_cleans_up_temp(self, tmp_path):
        # A file can never be os.replace'd onto an existing directory -- this
        # forces a real failure in the swap step without any monkeypatching.
        dest = tmp_path / 'dest'
        dest.mkdir()
        with pytest.raises(IsADirectoryError):
            atomic_write_text(dest, 'new\n')
        assert dest.is_dir()
        assert list(dest.iterdir()) == []
        assert list(tmp_path.iterdir()) == [dest]


class TestAtomicCopy:
    def test_copies_content_verbatim(self, tmp_path):
        src = tmp_path / 'src.geojson'
        src.write_text('{"a": 1}')
        dest = tmp_path / 'dest.geojson'
        atomic_copy(src, dest)
        assert dest.read_text() == '{"a": 1}'

    def test_replaces_existing_content(self, tmp_path):
        src = tmp_path / 'src.geojson'
        src.write_text('{"a": 2}')
        dest = tmp_path / 'dest.geojson'
        dest.write_text('{"a": 1}')
        atomic_copy(src, dest)
        assert dest.read_text() == '{"a": 2}'

    def test_leaves_no_temp_debris(self, tmp_path):
        src = tmp_path / 'src.geojson'
        src.write_text('{}')
        dest = tmp_path / 'dest.geojson'
        atomic_copy(src, dest)
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            'dest.geojson',
            'src.geojson',
        ]

    def test_failure_leaves_dest_unchanged_and_cleans_up_temp(self, tmp_path):
        missing_src = tmp_path / 'missing.geojson'
        dest = tmp_path / 'dest.geojson'
        dest.write_text('{"kept": true}')
        with pytest.raises(FileNotFoundError):
            atomic_copy(missing_src, dest)
        assert dest.read_text() == '{"kept": true}'
        assert [p.name for p in tmp_path.iterdir()] == ['dest.geojson']


class TestStagedDir:
    def test_populates_fresh_dest(self, tmp_path):
        dest = tmp_path / 'cogs' / '20240101'
        dest.parent.mkdir()
        with staged_dir(dest) as staging:
            (staging / 'a.tif').write_text('data')
        assert (dest / 'a.tif').read_text() == 'data'

    def test_replaces_existing_dir_wholesale(self, tmp_path):
        dest = tmp_path / 'cogs' / '20240101'
        dest.mkdir(parents=True)
        (dest / 'stale.tif').write_text('stale')
        with staged_dir(dest) as staging:
            (staging / 'fresh.tif').write_text('fresh')
        # The whole dir was swapped -- the stale member from the old dir is
        # gone, not merged with the new one.
        assert [p.name for p in dest.iterdir()] == ['fresh.tif']
        assert (dest / 'fresh.tif').read_text() == 'fresh'

    def test_failure_leaves_existing_dest_untouched(self, tmp_path):
        dest = tmp_path / 'cogs' / '20240101'
        dest.mkdir(parents=True)
        (dest / 'keep.tif').write_text('keep')
        with pytest.raises(RuntimeError, match='boom'), staged_dir(dest) as staging:
            _write_then_blow_up(staging)
        assert [p.name for p in dest.iterdir()] == ['keep.tif']
        assert (dest / 'keep.tif').read_text() == 'keep'

    def test_failure_when_dest_absent_leaves_nothing_behind(self, tmp_path):
        parent = tmp_path / 'cogs'
        parent.mkdir()
        dest = parent / '20240101'
        with pytest.raises(RuntimeError, match='boom'), staged_dir(dest) as staging:
            _write_then_blow_up(staging)
        assert not dest.exists()
        assert list(parent.iterdir()) == []

    def test_leaves_no_temp_or_old_debris_on_success(self, tmp_path):
        parent = tmp_path / 'cogs'
        parent.mkdir()
        dest = parent / '20240101'
        dest.mkdir()
        (dest / 'stale.tif').write_text('stale')
        with staged_dir(dest) as staging:
            (staging / 'fresh.tif').write_text('fresh')
        assert [p.name for p in parent.iterdir()] == ['20240101']
