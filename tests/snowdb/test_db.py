"""SnowDb discovery over the data/ directory, with injected specs."""

import pytest

from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.spec import DatasetSpec, GridParams


def _spec(name: str) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        grid_params=GridParams(
            origin_x=-120.0,
            origin_y=45.0,
            px_size=0.01,
            cols=256,
            rows=256,
            tile_size=256,
        ),
        dem_min_m=0.0,
        dem_max_m=1000.0,
    )


def test_discovers_configured_dataset(tmp_path):
    (tmp_path / 'data' / 'snodas').mkdir(parents=True)

    db = SnowDb(tmp_path, [_spec('snodas')])

    assert 'snodas' in db
    assert list(db) == ['snodas']
    assert db['snodas'].spec.name == 'snodas'
    assert db['snodas'].path == tmp_path / 'data' / 'snodas'


def test_skips_dotfiles_and_stray_files(tmp_path):
    data = tmp_path / 'data'
    (data / 'snodas').mkdir(parents=True)
    (data / '.DS_Store').write_text('')  # hidden -> skipped
    (data / 'README.txt').write_text('')  # non-dir -> skipped

    db = SnowDb(tmp_path, [_spec('snodas')])

    assert list(db) == ['snodas']


def test_unknown_dataset_dir_raises(tmp_path):
    (tmp_path / 'data' / 'mystery').mkdir(parents=True)

    with pytest.raises(ValueError, match='Unknown dataset directory'):
        SnowDb(tmp_path, [_spec('snodas')])


def test_missing_data_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match='No data directory'):
        SnowDb(tmp_path, [_spec('snodas')])


def test_duplicate_spec_names_rejected(tmp_path):
    (tmp_path / 'data').mkdir()

    with pytest.raises(ValueError, match='Duplicate dataset spec name'):
        SnowDb(tmp_path, [_spec('snodas'), _spec('snodas')])


def test_default_specs_discover_snodas(tmp_path):
    """The built-in DEFAULT_DATASET_SPECS wires up the real snodas spec."""
    from snowtool.snowdb.datasets import DEFAULT_DATASET_SPECS

    (tmp_path / 'data' / 'snodas').mkdir(parents=True)

    db = SnowDb(tmp_path, DEFAULT_DATASET_SPECS)

    assert db['snodas'].spec.name == 'snodas'
