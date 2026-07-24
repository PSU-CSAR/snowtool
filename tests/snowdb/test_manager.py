"""SnowDbManager: initialize/stage/register/activate/resolve/create_dataset."""

import asyncio
import shutil

from datetime import date
from pathlib import Path

import pytest

from snowtool.exceptions import SnowDbConfigError
from snowtool.snowdb.config import (
    CONFIG_FILENAME,
    DATASET_CONFIG_FILENAME,
    PathDatasetLink,
    RootConfig,
)
from snowtool.snowdb.coverage import Coverage
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.datasets import config_from_spec
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.manager import SnowDbManager
from snowtool.snowdb.pourpoint import Pourpoint
from snowtool.snowdb.query import DateRangeQuery
from snowtool.snowdb.reader import SnowDbReader
from snowtool.snowdb.spec import DatasetSpec, GridParams

from ..conftest import (
    SWE_VALUE,
    CapturingProgress,
    make_manager,
    register_dataset_config,
    write_swe_cog,
)


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


def test_initialize_creates_the_base_layout(tmp_path):
    SnowDbManager.initialize(tmp_path)

    assert (tmp_path / 'pourpoints').is_dir()
    assert (tmp_path / 'pourpoints' / 'records').is_dir()
    assert (tmp_path / 'data').is_dir()


def test_initialize_writes_a_loadable_root_config(tmp_path):
    SnowDbManager.initialize(tmp_path)

    config = RootConfig.load(tmp_path / CONFIG_FILENAME)
    # No datasets are registered by init -- a dataset exists only once its link
    # is registered (and is served only while that link is active), not by being
    # a configured spec.
    assert config.resource == 'snowtool.snowdb/v1'
    assert config.datasets == {}


def test_initialize_preserves_an_existing_config(tmp_path):
    SnowDbManager.initialize(tmp_path)
    config_path = tmp_path / CONFIG_FILENAME
    created_at = RootConfig.load(config_path).created_at

    SnowDbManager.initialize(tmp_path)  # idempotent re-init

    assert RootConfig.load(config_path).created_at == created_at


def test_initialize_is_idempotent(tmp_path):
    SnowDbManager.initialize(tmp_path)
    # A second init against the same root must not raise.
    SnowDbManager.initialize(tmp_path)

    assert (tmp_path / 'data').is_dir()


def test_rasterize_aoi_burns_every_active_dataset(
    tmp_path,
    spec,
    pourpoint_geojson,
):
    """A global AOI is rasterized once per active dataset, on each one's grid."""
    spec_b = DatasetSpec(
        name='snodas',
        grid_params=spec.grid_params,
    )
    # `spec` (name='test') and `spec_b` (name='snodas') share the synthetic grid.
    data = tmp_path / 'data'
    data.mkdir()
    Dataset.create(spec, data / spec.name)  # skeleton only; return unused
    Dataset.create(spec_b, data / spec_b.name)

    manager = make_manager(tmp_path, [spec, spec_b])
    pourpoint = Pourpoint.from_geojson(pourpoint_geojson)
    result = manager.pourpoints.rasterize_aois(
        [pourpoint],
        list(manager.db.registered.values()),
    )

    assert set(manager.db.registered) == {'test', 'snodas'}
    assert set(result.built) == {
        (pourpoint.station_triplet, 'test'),
        (pourpoint.station_triplet, 'snodas'),
    }
    assert result.skipped == []
    for name, dataset in manager.db.registered.items():
        raster_path = dataset.aoi_raster_path_from_triplet(pourpoint.station_triplet)
        assert raster_path.exists()
        assert raster_path.parent == data / name / 'aoi-rasters'


def test_dataset_create_is_idempotent(tmp_path, spec):
    # Dataset.create converges: a first call builds the skeleton and reports
    # created=True; a re-run over the existing skeleton reports created=False and
    # does not raise (no refuse-to-clobber). It also converges a *partial*
    # skeleton (one subdir present) into the full one, still reporting created.
    path = tmp_path / 'db'

    ds, created = Dataset.create(spec, path)
    assert created is True
    assert ds._aoi_rasters.is_dir()
    assert ds._cogs.is_dir()

    _, again = Dataset.create(spec, path)
    assert again is False

    # A partial skeleton (cogs/ removed) is still "not fully present" -> created,
    # and the missing half is rebuilt without clobbering the surviving half.
    shutil.rmtree(ds._cogs)
    _, repaired = Dataset.create(spec, path)
    assert repaired is True
    assert ds._aoi_rasters.is_dir()
    assert ds._cogs.is_dir()


def test_rasterize_aoi_creates_a_missing_aoi_rasters_dir(dataset, pourpoint_geojson):
    # A dataset with no data on disk yet (here: its aoi-rasters/ dir removed)
    # still rasterizes -- the write path recreates the dataset subdir.
    shutil.rmtree(dataset._aoi_rasters)
    assert not dataset._aoi_rasters.exists()

    pourpoint = Pourpoint.from_geojson(pourpoint_geojson)
    assert dataset.rasterize_aoi(pourpoint)

    assert dataset._aoi_rasters.is_dir()
    assert dataset.load_aoi_raster(pourpoint.station_triplet).path.exists()


# --- staged dataset registration (3c) ----------------------------------------


def test_staged_dataset_registration_end_to_end(tmp_path, spec, pourpoint_geojson):
    # Import a pourpoint first, then stage + commit a dataset over it. Staging is
    # invisible to readers; the config write is the commit point.
    SnowDbManager.initialize(tmp_path)
    manager = SnowDbManager.open(tmp_path)
    manager.pourpoints.import_(pourpoint_geojson)

    config = config_from_spec(spec)
    ds_dir = manager.db.dataset_dir('test', config)
    ds_dir.mkdir(parents=True, exist_ok=True)
    config_path = ds_dir / DATASET_CONFIG_FILENAME
    config.save(config_path)

    staged = manager.stage_dataset('test', config_path)

    assert staged.created is True
    assert staged.coverage == {'12345:MT:USGS': Coverage.FULL}
    assert staged.rasterized.built == [('12345:MT:USGS', 'test')]
    assert (ds_dir / 'aoi-rasters' / '12345_MT_USGS.tif').is_file()

    # Re-staging converges: the skeleton is tolerated and the provenance-current
    # AOI raster is skipped, not rebuilt (no implicit force -- a byte-level
    # rebuild is `rasterize_aois(rebuild=True)` / `pourpoint rasterize --rebuild`).
    progress = CapturingProgress()
    restaged = manager.stage_dataset('test', config_path, progress=progress)
    assert restaged.created is False
    assert restaged.rasterized.built == []
    assert restaged.rasterized.skipped == [('12345:MT:USGS', 'test')]
    assert restaged.coverage == staged.coverage
    # Staging reports each slow phase sequentially: record parse, then AOI
    # rasterize (per pourpoint x dataset pair). Coverage is no longer a separate
    # phase -- the rasterize pass computes it once for its own Coverage.NONE skip
    # and surfaces it on the result.
    assert [(t.label, t.total, t.advanced) for t in progress.tasks] == [
        ('parsing 1 pourpoint record(s)', 1, 1),
        ('rasterizing', 1, 1),
    ]

    # Before commit: a fresh open does NOT see the dataset (config unwritten).
    assert list(SnowDb.open(tmp_path)) == []

    manager.register_dataset('test', config_path, coverage=staged.coverage)

    reopened = SnowDb.open(tmp_path)
    assert list(reopened) == ['test']
    assert reopened.pourpoint_dataset_coverage('12345:MT:USGS', 'test') is Coverage.FULL

    # Working stats: ingest a uniform SWE COG over the covered basin and reduce it.
    ds = reopened['test']
    write_swe_cog(ds)
    stats = asyncio.run(
        SnowDbReader(reopened).zonal_stats(
            '12345:MT:USGS',
            'test',
            DateRangeQuery(start_date=date(2018, 4, 27), end_date=date(2018, 4, 27)),
            variable_keys=['swe'],
        ),
    )
    compact = stats.dump_compact()
    (matrix,) = compact.results.values()
    assert matrix == [[pytest.approx(SWE_VALUE)]]


def test_create_dataset_stages_and_registers_inactive(tmp_path, spec):
    # create_dataset owns the whole stamp-a-new-dataset lifecycle: resolve the
    # directory, write the config, stage, register inactive.
    SnowDbManager.initialize(tmp_path)
    manager = SnowDbManager.open(tmp_path)

    result = manager.create_dataset('test', config_from_spec(spec))

    assert result.staged.created is True
    assert result.registered is True
    assert result.staged.dataset.path == tmp_path / 'data' / 'test'
    # The config was written beside its data ...
    assert (tmp_path / 'data' / 'test' / DATASET_CONFIG_FILENAME).is_file()
    # ... and it was registered inactive: it exists but readers ignore it.
    config = RootConfig.load(tmp_path / CONFIG_FILENAME)
    assert config.datasets['test'].active is False
    assert list(SnowDb.open(tmp_path)) == []


def test_create_dataset_reregister_preserves_active_link(tmp_path, spec):
    # The one real invariant: an existing registration is never clobbered. Once a
    # dataset is active, a re-create leaves its link and active flag untouched and
    # reports registered=False. A fresh manager per call mirrors the CLI (each
    # process reopens the root config).
    SnowDbManager.initialize(tmp_path)
    config = config_from_spec(spec)

    first = SnowDbManager.open(tmp_path).create_dataset('test', config)
    assert first.registered is True

    SnowDbManager.open(tmp_path).set_dataset_active('test', True)

    second = SnowDbManager.open(tmp_path).create_dataset('test', config)
    assert second.registered is False
    assert second.staged.created is False
    # The active flag survives the re-create verbatim.
    assert RootConfig.load(tmp_path / CONFIG_FILENAME).datasets['test'].active is True


def test_create_dataset_does_not_clobber_a_same_manager_registration(tmp_path, spec):
    # The no-clobber invariant must hold within ONE manager instance too: a
    # register_dataset then create_dataset on the *same* manager. self.db is a
    # snapshot frozen at construction (no write refreshes it), so create_dataset
    # checks the on-disk root config, not self.db.registered -- otherwise the
    # just-registered active link would be silently re-registered inactive.
    manager = SnowDbManager.initialize(tmp_path)
    config = config_from_spec(spec)
    config_path = register_dataset_config(manager, 'test', config, active=True)

    result = manager.create_dataset('test', config)

    # It saw the existing registration despite the stale snapshot: no re-register.
    assert result.registered is False
    on_disk = RootConfig.load(tmp_path / CONFIG_FILENAME).datasets['test']
    assert on_disk.active is True
    assert on_disk == PathDatasetLink(
        path=Path('data/test/dataset.json'),
        active=True,
    )
    # The link still points where register wrote it (not relinked out from under
    # readers). config_path is the on-disk config create_dataset would have used.
    assert config_path == tmp_path / 'data' / 'test' / DATASET_CONFIG_FILENAME


def test_create_dataset_absolute_data_dir_round_trips_through_open(tmp_path, spec):
    # The single-rule invariant, in the form that survives create: an absolute
    # data_dir points create's config write and a later SnowDb.open at the SAME
    # directory. create writes the config there; open resolves the dataset's data
    # to that exact path (not a nested <dir>/<dir>).
    SnowDbManager.initialize(tmp_path)
    manager = SnowDbManager.open(tmp_path)
    data_dir = tmp_path / 'elsewhere' / 'test-data'
    config = config_from_spec(spec).model_copy(update={'data_dir': data_dir})

    result = manager.create_dataset('test', config)

    # create wrote the config beside its data at the absolute data_dir ...
    assert result.staged.dataset.path == data_dir
    assert (data_dir / DATASET_CONFIG_FILENAME).is_file()
    # ... and a fresh open resolves the dataset's data to that same directory.
    reopened = SnowDb.open(tmp_path)
    assert reopened.registered['test'].path == data_dir


def test_create_dataset_rejects_a_relative_data_dir(tmp_path, spec):
    # create cannot honor a relative data_dir: it writes the config *at* the
    # directory data_dir names, but SnowDb.open later resolves a relative data_dir
    # against that config's own dir, so the two would disagree (<root>/<dir> vs.
    # <root>/<dir>/<dir>). Refuse it up front with a typed, actionable error and
    # write nothing.
    SnowDbManager.initialize(tmp_path)
    manager = SnowDbManager.open(tmp_path)
    config = config_from_spec(spec).model_copy(update={'data_dir': Path('nested')})

    with pytest.raises(SnowDbConfigError) as excinfo:
        manager.create_dataset('test', config)

    # The message says what to pass instead (omit for the convention, or absolute).
    message = str(excinfo.value)
    assert 'relative data_dir' in message
    assert 'absolute' in message
    # Nothing was written or registered: neither the errant nested dir nor a
    # root-relative one holds a config, and the root config stayed empty.
    assert not (tmp_path / 'nested').exists()
    assert not (tmp_path / 'nested' / 'nested').exists()
    assert RootConfig.load(tmp_path / CONFIG_FILENAME).datasets == {}


def test_resolve_dataset_partitions_paths_from_names(tmp_path, spec):
    # The token partition is syntactic: a separator/.json token is a path (the
    # catalog is never consulted); a bare token is a NAME resolved only from the
    # root config -- an unregistered staged dataset no longer resolves by the
    # old data/<name>/dataset.json convention.
    manager = SnowDbManager.initialize(tmp_path)
    staged_dir = tmp_path / 'data' / 'staged'
    staged_dir.mkdir(parents=True)
    config_path = staged_dir / DATASET_CONFIG_FILENAME
    config_from_spec(spec).save(config_path)

    # Not linked in the root config -> a reader does not serve it ...
    assert 'staged' not in manager.db.datasets
    # ... and its explicit config path resolves (name from the parent dir) ...
    assert manager.resolve_dataset(str(config_path)).spec.name == 'staged'
    # ... but the bare NAME does not: names never probe the filesystem.
    with pytest.raises(ValueError, match="No such dataset 'staged'"):
        manager.resolve_dataset('staged')

    with pytest.raises(ValueError, match="No such dataset 'nope'"):
        manager.resolve_dataset('nope')


def test_resolve_dataset_name_is_never_shadowed_by_a_file(tmp_path, monkeypatch, spec):
    # Collision-proofing: with a file in cwd named exactly like a registered
    # dataset, the bare token still resolves the REGISTERED dataset (a name
    # never touches the filesystem), and the path spelling of the same token
    # never resolves the catalog entry.
    manager = SnowDbManager.initialize(tmp_path)
    register_dataset_config(manager, 'test', config_from_spec(spec))
    manager = SnowDbManager.open(tmp_path)

    workdir = tmp_path / 'work'
    workdir.mkdir()
    (workdir / 'test').write_text('{}')  # not a dataset config
    (workdir / 'ghost').write_text('{}')  # no such registered dataset
    monkeypatch.chdir(workdir)

    resolved = manager.resolve_dataset('test')
    assert resolved.spec.name == 'test'
    assert resolved.path == tmp_path / 'data' / 'test'

    # './test' has a separator -> the path branch: the file exists but is not a
    # dataset config, so it fails as a clean SnowDbConfigError instead of falling
    # back to the name (or leaking a raw pydantic ValidationError).
    with pytest.raises(SnowDbConfigError, match='not a usable dataset config'):
        manager.resolve_dataset('./test')

    # An unregistered bare name raises even though a file of that name exists.
    with pytest.raises(ValueError, match="No such dataset 'ghost'"):
        manager.resolve_dataset('ghost')


def test_resolve_dataset_path_token_must_exist(tmp_path):
    manager = SnowDbManager.initialize(tmp_path)

    with pytest.raises(ValueError, match='No dataset config file at'):
        manager.resolve_dataset('data/nope/dataset.json')


@pytest.mark.parametrize('name', ['a/b', 'a\\b', 'x.json'])
def test_register_dataset_rejects_pathlike_names(tmp_path, name):
    # A name must be usable as a bare resolve_dataset token and a directory
    # name; registration is the single choke point that enforces it.
    manager = SnowDbManager.initialize(tmp_path)

    with pytest.raises(ValueError, match='Invalid dataset name'):
        manager.register_dataset(name, tmp_path / DATASET_CONFIG_FILENAME)

    # The rejected call wrote nothing -- the config still has no datasets.
    assert RootConfig.load(tmp_path / CONFIG_FILENAME).datasets == {}


# --- register/activate split ---------------------------------------------------


def test_inactive_dataset_is_registered_but_not_served(tmp_path, spec):
    manager = SnowDbManager.initialize(tmp_path)
    register_dataset_config(manager, spec.name, config_from_spec(spec), active=False)

    db = SnowDb.open(tmp_path)
    # Registered (the management surface binds it) but not served (readers skip it).
    assert spec.name in db.registered
    assert spec.name not in db.datasets
    assert list(db) == []

    # set_dataset_active flips the flag; a reopen serves the dataset.
    SnowDbManager(db).set_dataset_active(spec.name, True)
    reopened = SnowDb.open(tmp_path)
    assert list(reopened) == [spec.name]
    assert reopened.datasets[spec.name] is reopened.registered[spec.name]


def test_set_dataset_active_rejects_an_unregistered_name(tmp_path):
    manager = SnowDbManager.initialize(tmp_path)

    with pytest.raises(ValueError, match="No registered dataset 'nope'"):
        manager.set_dataset_active('nope', True)


# --- register_dataset error paths (WS6) ---------------------------------------


def test_register_dataset_rejects_a_malformed_linked_config(tmp_path):
    # A config that exists but doesn't parse/resolve is caught before the write,
    # not deferred to the next reader open -- no caller can commit a broken link.
    manager = SnowDbManager.initialize(tmp_path)
    root_config_path = tmp_path / CONFIG_FILENAME
    before = root_config_path.read_bytes()

    bad = tmp_path / 'bad.json'
    bad.write_text('{"resource": "snowtool.dataset/v1", "grid": {}, "variables": {}}')

    with pytest.raises(SnowDbConfigError, match='not a usable dataset config'):
        manager.register_dataset('test', bad)

    # Nothing was written -- the root config on disk is byte-for-byte unchanged.
    assert root_config_path.read_bytes() == before
    assert RootConfig.load(root_config_path).datasets == {}


def test_register_dataset_nonexistent_config_path_defers_to_open(tmp_path):
    # register_dataset is the commit point for the *link*, not a read of the
    # config it points at -- so a path that was never created is accepted here
    # and only surfaces (as the existing "dangling link" SnowDbConfigError) when
    # a reader actually opens the database.
    manager = SnowDbManager.initialize(tmp_path)
    bogus = tmp_path / 'data' / 'test' / DATASET_CONFIG_FILENAME
    assert not bogus.is_file()

    manager.register_dataset('test', bogus)

    loaded = RootConfig.load(tmp_path / CONFIG_FILENAME)
    assert 'test' in loaded.datasets
    with pytest.raises(SnowDbConfigError, match='missing config'):
        SnowDb.open(tmp_path)


def test_register_dataset_overwrites_an_existing_link(tmp_path, spec):
    # Per the docstring: re-registering a name overwrites its link outright
    # (no merge, no error) -- the last registration wins.
    manager = SnowDbManager.initialize(tmp_path)
    config = config_from_spec(spec)

    first_dir = tmp_path / 'first'
    first_dir.mkdir()
    first_path = first_dir / DATASET_CONFIG_FILENAME
    config.save(first_path)
    manager.register_dataset(spec.name, first_path)

    second_dir = tmp_path / 'second'
    second_dir.mkdir()
    second_path = second_dir / DATASET_CONFIG_FILENAME
    config.save(second_path)
    manager.register_dataset(spec.name, second_path)

    link = RootConfig.load(tmp_path / CONFIG_FILENAME).datasets[spec.name]
    assert isinstance(link, PathDatasetLink)
    # Both stage dirs live under the root, so the link is stored relative.
    assert link.path == Path('second/dataset.json')

    # A fresh open resolves the dataset beside its *second* config, not the first.
    opened = SnowDb.open(tmp_path)
    assert opened[spec.name].path == second_dir
