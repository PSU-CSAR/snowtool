import pytest

from click.testing import CliRunner

from snowtool.cli import cli
from snowtool.migration.restructure import restructure_to_snowdb


def _make_flat(src):
    (src / 'aoi-rasters').mkdir(parents=True)
    (src / 'cogs').mkdir()
    (src / 'areas.tif').write_text('areas')
    (src / 'dem.tif').write_text('dem')


def test_restructure_moves_flat_dir_into_layout(tmp_path):
    src = tmp_path / 'rasterdb'
    _make_flat(src)
    dst = tmp_path / 'snowdb'

    dataset_dir = restructure_to_snowdb(src, dst, 'snodas')

    assert dataset_dir == dst / 'data' / 'snodas'
    assert (dst / 'data' / 'snodas' / 'areas.tif').read_text() == 'areas'
    assert (dst / 'data' / 'snodas' / 'aoi-rasters').is_dir()
    assert (dst / 'data' / 'snodas' / 'cogs').is_dir()
    assert (dst / 'aois').is_dir()
    assert not src.exists()  # moved, not copied


def test_restructure_refuses_existing_destination(tmp_path):
    src = tmp_path / 'rasterdb'
    src.mkdir()
    dst = tmp_path / 'snowdb'
    (dst / 'data' / 'snodas').mkdir(parents=True)

    with pytest.raises(FileExistsError, match='already exists'):
        restructure_to_snowdb(src, dst, 'snodas')


def test_restructure_rejects_non_directory_source(tmp_path):
    with pytest.raises(ValueError, match='not a directory'):
        restructure_to_snowdb(tmp_path / 'missing', tmp_path / 'dst', 'snodas')


def test_cli_migration_restructure(tmp_path):

    src = tmp_path / 'rasterdb'
    _make_flat(src)
    dst = tmp_path / 'snowdb'

    result = CliRunner().invoke(
        cli,
        ['migration', 'restructure', str(src), str(dst)],
    )

    assert result.exit_code == 0, result.output
    assert (dst / 'data' / 'snodas' / 'dem.tif').exists()
