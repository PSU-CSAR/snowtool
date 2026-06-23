"""SnowDb binds its configured specs; SnowDbManager creates/registers/rasterizes."""

import shutil

import pytest

from snowtool.exceptions import SnowDbConfigError
from snowtool.snowdb.aoi import AOI
from snowtool.snowdb.config import CONFIG_FILENAME, RootConfig
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.manager import SnowDbManager
from snowtool.snowdb.spec import DatasetSpec, GridParams

from ..conftest import make_manager, make_snowdb


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
    )


def test_binds_configured_dataset(tmp_path):
    (tmp_path / 'data' / 'snodas').mkdir(parents=True)

    db = make_snowdb(tmp_path, [_spec('snodas')])

    assert 'snodas' in db
    assert list(db) == ['snodas']
    assert db['snodas'].spec.name == 'snodas'
    assert db['snodas'].path == tmp_path / 'data' / 'snodas'


def test_binds_every_spec_without_data_on_disk(tmp_path):
    # Datasets come from the configured specs, not from what's on disk, so a
    # spec is bound (to its would-be data/<name>/ dir) even on an empty root.
    db = make_snowdb(tmp_path, [_spec('snodas')])

    assert list(db) == ['snodas']
    assert db['snodas'].path == tmp_path / 'data' / 'snodas'


def test_missing_dirs_logs_a_warning(tmp_path, caplog):
    with caplog.at_level('WARNING'):
        make_snowdb(tmp_path, [_spec('snodas')])

    assert 'snowdb init' in caplog.text


def test_no_warning_on_an_initialized_root(tmp_path, caplog):
    SnowDbManager.initialize(tmp_path, [_spec('snodas')])

    with caplog.at_level('WARNING'):
        make_snowdb(tmp_path, [_spec('snodas')])

    assert caplog.text == ''


def test_initialize_creates_the_base_layout(tmp_path):
    SnowDbManager.initialize(tmp_path, [_spec('snodas')])

    assert (tmp_path / 'aois').is_dir()
    assert (tmp_path / 'data').is_dir()
    assert (tmp_path / 'data' / 'snodas').is_dir()


def _register(manager: SnowDbManager, spec: DatasetSpec) -> None:
    """Stage ``spec`` as an on-disk dataset config and register its link."""
    from snowtool.snowdb.config import DATASET_CONFIG_FILENAME
    from snowtool.snowdb.datasets import config_from_spec

    ds_dir = manager.db.data_path / spec.name
    ds_dir.mkdir(parents=True, exist_ok=True)
    config_path = ds_dir / DATASET_CONFIG_FILENAME
    config_from_spec(spec).save(config_path)
    manager.register_dataset(spec.name, config_path)


def test_initialize_writes_a_loadable_root_config(tmp_path):
    SnowDbManager.initialize(tmp_path, [_spec('snodas')])

    config = RootConfig.load(tmp_path / CONFIG_FILENAME)
    # No datasets are registered by init -- a dataset goes live only by adding its
    # link, not by being a configured spec.
    assert config.resource == 'snowtool.snowdb/v1'
    assert config.datasets == {}


def test_initialize_preserves_an_existing_config(tmp_path):
    SnowDbManager.initialize(tmp_path, [_spec('snodas')])
    config_path = tmp_path / CONFIG_FILENAME
    created_at = RootConfig.load(config_path).created_at

    SnowDbManager.initialize(tmp_path, [_spec('snodas')])  # idempotent re-init

    assert RootConfig.load(config_path).created_at == created_at


def test_open_requires_a_root_config(tmp_path):
    # A bare directory (no snowdb_conf.json) is not a snowdb open will serve.
    with pytest.raises(SnowDbConfigError, match='migration stamp'):
        SnowDb.open(tmp_path)


def test_open_sees_no_datasets_after_bare_init(tmp_path):
    SnowDbManager.initialize(tmp_path, [_spec('snodas')])

    # init registers nothing, so open (which follows links) binds no datasets even
    # though a data/<name>/ dir was staged.
    assert list(SnowDb.open(tmp_path)) == []


def test_open_binds_registered_datasets(tmp_path):
    manager = SnowDbManager.initialize(tmp_path, [_spec('snodas')])
    _register(manager, _spec('snodas'))

    assert list(SnowDb.open(tmp_path)) == ['snodas']


def test_open_accepts_the_config_file_directly(tmp_path):
    manager = SnowDbManager.initialize(tmp_path, [_spec('snodas')])
    _register(manager, _spec('snodas'))

    opened = SnowDb.open(tmp_path / CONFIG_FILENAME)

    assert opened.path == tmp_path
    assert list(opened) == ['snodas']


def test_open_errors_on_a_dangling_link(tmp_path):
    manager = SnowDbManager.initialize(tmp_path, [_spec('snodas')])
    _register(manager, _spec('snodas'))
    # Remove the linked config out of band -> open must fail cleanly.
    (manager.db.data_path / 'snodas' / 'dataset.json').unlink()

    with pytest.raises(SnowDbConfigError, match='missing config'):
        SnowDb.open(tmp_path)


def test_initialize_is_idempotent(tmp_path):
    SnowDbManager.initialize(tmp_path, [_spec('snodas')])
    # A second init against the same root must not raise.
    SnowDbManager.initialize(tmp_path, [_spec('snodas')])

    assert (tmp_path / 'data' / 'snodas').is_dir()


def test_require_initialized_raises_on_uninitialized_root(tmp_path):
    manager = make_manager(tmp_path, [_spec('snodas')])

    with pytest.raises(FileNotFoundError, match='not an initialized snowdb'):
        manager.require_initialized()


def test_require_initialized_passes_after_init(tmp_path):
    manager = SnowDbManager.initialize(tmp_path, [_spec('snodas')])

    assert manager.require_initialized() is manager


def test_duplicate_spec_names_rejected():
    # A root config keys datasets by name, so duplicates can't be expressed in
    # one; the constructor's guard is asserted directly here.
    with pytest.raises(ValueError, match='Duplicate dataset spec name'):
        SnowDb._index_specs([_spec('snodas'), _spec('snodas')])


def test_specs_colliding_on_model_name_rejected(tmp_path):
    # 'foo-bar' and 'foo_bar' are distinct dataset names but both generate the
    # response-model prefix 'FooBar', which would collide in the OpenAPI schema.
    with pytest.raises(ValueError, match='same response-model name'):
        make_snowdb(tmp_path, [_spec('foo-bar'), _spec('foo_bar')])


def test_rasterize_aoi_burns_every_active_dataset(
    tmp_path,
    spec,
    source_dem,
    aoi_geojson,
):
    """A global AOI is rasterized once per active dataset, on each one's grid."""
    spec_b = DatasetSpec(
        name='snodas',
        grid_params=spec.grid_params,
    )
    # `spec` (name='test') and `spec_b` (name='snodas') share the synthetic grid,
    # so the one source DEM covers both.
    data = tmp_path / 'data'
    data.mkdir()
    Dataset.create(spec, data / spec.name, source_dem)
    Dataset.create(spec_b, data / spec_b.name, source_dem)

    manager = make_manager(tmp_path, [spec, spec_b])
    rasters = manager.rasterize_aoi(AOI.from_geojson(aoi_geojson))

    assert set(rasters) == {'test', 'snodas'}
    for name, raster in rasters.items():
        assert raster.path.exists()
        assert raster.path.parent == data / name / 'aoi-rasters'


def test_rasterize_aoi_creates_a_missing_aoi_rasters_dir(dataset, aoi_geojson):
    # A dataset with no data on disk yet (here: its aoi-rasters/ dir removed)
    # still rasterizes -- the write path recreates the dataset subdir.
    shutil.rmtree(dataset._aoi_rasters)
    assert not dataset._aoi_rasters.exists()

    raster = dataset.rasterize_aoi(AOI.from_geojson(aoi_geojson))

    assert dataset._aoi_rasters.is_dir()
    assert raster.path.exists()


def test_default_specs_bind_snodas(tmp_path):
    """The built-in DEFAULT_DATASET_SPECS wires up the real snodas spec."""
    from snowtool.snowdb.datasets import DEFAULT_DATASET_SPECS

    db = make_snowdb(tmp_path, DEFAULT_DATASET_SPECS)

    assert db['snodas'].spec.name == 'snodas'


def test_aoi_paths_empty_without_records_dir(tmp_path):
    db = make_snowdb(tmp_path, [_spec('snodas')])

    assert db.aoi_paths() == []


def test_aoi_paths_lists_and_sorts_geojson(tmp_path, aoi_geojson):
    db = SnowDbManager.initialize(tmp_path, [_spec('snodas')]).db
    shutil.copy(aoi_geojson, db.aoi_records_path / 'b.geojson')
    shutil.copy(aoi_geojson, db.aoi_records_path / 'a.geojson')
    # A non-geojson file is ignored.
    (db.aoi_records_path / 'notes.txt').write_text('x')

    assert db.aoi_paths() == [
        db.aoi_records_path / 'a.geojson',
        db.aoi_records_path / 'b.geojson',
    ]


def test_aois_parse_global_geojson(tmp_path, aoi_geojson):
    db = SnowDbManager.initialize(tmp_path, [_spec('snodas')]).db
    shutil.copy(aoi_geojson, db.aoi_records_path / 'pourpoint.geojson')

    aois = list(db.aois())

    assert len(aois) == 1
    assert aois[0].station_triplet == '12345:MT:USGS'


def test_aoi_triplets(tmp_path, aoi_geojson):
    db = SnowDbManager.initialize(tmp_path, [_spec('snodas')]).db
    shutil.copy(aoi_geojson, db.aoi_records_path / 'pourpoint.geojson')

    assert db.aoi_triplets() == {'12345:MT:USGS'}
