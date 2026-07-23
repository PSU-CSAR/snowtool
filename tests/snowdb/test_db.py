"""SnowDb binds its configured specs; SnowDbManager creates/registers/rasterizes."""

import asyncio
import json
import os
import shutil

from datetime import date
from pathlib import Path

import pytest

from snowtool.exceptions import (
    IndexedPourpointMissingBasinError,
    PourpointNotFoundError,
    SnowDbConfigError,
    UnknownDatasetError,
)
from snowtool.snowdb.config import (
    CONFIG_FILENAME,
    DATASET_CONFIG_FILENAME,
    InlineDatasetLink,
    PathDatasetLink,
    RootConfig,
)
from snowtool.snowdb.coverage import Coverage
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.datasets import DEFAULT_DATASET_SPECS, config_from_spec
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
    make_snowdb,
    make_spec,
    register_dataset_config,
    write_pourpoint_record,
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


def test_in_code_config_with_absolute_paths_opens_without_root(tmp_path):
    # A config built in code (never saved -> no root) opens as long as its links
    # are absolute: here an inline dataset with an absolute data_dir.

    data_dir = tmp_path / 'anywhere' / 'snodas'
    dataset_config = config_from_spec(_spec('snodas'))
    dataset_config.data_dir = data_dir
    config = RootConfig.create()
    config.pourpoint_records = str(tmp_path / 'pourpoints' / 'records')
    config.pourpoint_index = str(tmp_path / 'pourpoints' / 'index.geojson')
    config.datasets['snodas'] = InlineDatasetLink(dataset=dataset_config)

    db = SnowDb(config)  # no config.path set -> no root

    assert db.root is None
    assert db['snodas'].path == data_dir


def test_inline_link_with_unknown_ingester_is_a_config_error(tmp_path):
    # The inline-link branch has no path to load, so it resolves the carried
    # config directly through from_config -- which raises a bare ValueError for an
    # unknown ingester. SnowDb.__init__ must wrap that into a SnowDbConfigError
    # (naming the root), exactly as the path-link branch does via load_dataset_spec.
    dataset_config = config_from_spec(_spec('snodas')).model_copy(
        update={'ingester': 'nope'},
    )
    config = RootConfig.create()
    config.pourpoint_records = str(tmp_path / 'pourpoints' / 'records')
    config.pourpoint_index = str(tmp_path / 'pourpoints' / 'index.geojson')
    config.datasets['snodas'] = InlineDatasetLink(dataset=dataset_config)

    with pytest.raises(
        SnowDbConfigError,
        match="inline dataset 'snodas' is not usable",
    ):
        SnowDb(config)


def test_relative_path_without_root_raises(tmp_path):
    # The default pourpoint_records is relative; with no root it cannot resolve, so
    # construction fails with a precise error rather than a silent default.
    config = RootConfig.create()

    with pytest.raises(SnowDbConfigError, match='no location'):
        SnowDb(config)


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


def test_open_requires_a_root_config(tmp_path):
    # A bare directory (no snowdb_conf.json) is not a snowdb open will serve.
    with pytest.raises(SnowDbConfigError, match='snowtool init'):
        SnowDb.open(tmp_path)


def test_open_malformed_root_config_is_a_config_error(tmp_path):
    # A file exists but doesn't parse/validate as a root config is always a clean
    # SnowDbConfigError (which the CLI renders), never a raw pydantic
    # ValidationError or UnicodeDecodeError. The 3-way taxonomy (truncated JSON,
    # non-utf-8 bytes, valid-JSON-wrong-shape) is pinned at the loader level in
    # test_config.py; here one representative case proves the wrap surfaces
    # through SnowDb.open.
    (tmp_path / CONFIG_FILENAME).write_bytes(b'{ "resource": "snowtool.snowdb/v1"')
    with pytest.raises(SnowDbConfigError, match='not a readable snowdb root config'):
        SnowDb.open(tmp_path)


def test_open_malformed_linked_dataset_config_is_a_config_error(tmp_path):
    # The same wrap, but in a *linked* dataset config rather than the root: still a
    # clean SnowDbConfigError naming the offending config file (raised by the
    # canonical DatasetConfig.load). The 3-way taxonomy is pinned at the loader
    # level in test_dataset_config.py; here one representative case proves it
    # surfaces through SnowDb.open.
    manager = SnowDbManager.initialize(tmp_path)
    register_dataset_config(manager, 'snodas', config_from_spec(_spec('snodas')))
    linked = manager.db.data_path / 'snodas' / 'dataset.json'
    linked.write_bytes(b'{ "resource": "snowtool.dataset/v1", "grid": {')

    with pytest.raises(SnowDbConfigError, match='not a usable dataset config'):
        SnowDb.open(tmp_path)


def test_open_linked_config_with_unknown_ingester_is_a_config_error(tmp_path):
    # A linked config that parses fine but names an ingester that does not
    # resolve is *unusable*, not merely malformed: DatasetSpec.from_config raises
    # a bare ValueError for it. SnowDb.open must wrap that into the same clean
    # SnowDbConfigError a malformed config gets (the canonical loader does the
    # wrapping), rather than leaking an untyped ValueError from the read path at
    # every CLI invocation / API startup.
    manager = SnowDbManager.initialize(tmp_path)
    config = config_from_spec(_spec('snodas'))
    linked = register_dataset_config(manager, 'snodas', config)
    # Rewrite the linked config (already committed) with an unknown ingester so
    # the failure lands at open, not at register time.
    config.model_copy(update={'ingester': 'nope'}).save(linked)

    with pytest.raises(SnowDbConfigError, match='not a usable dataset config'):
        SnowDb.open(tmp_path)


def test_open_sees_no_datasets_after_bare_init(tmp_path):
    SnowDbManager.initialize(tmp_path)

    # init registers nothing, so open (which follows links) binds no datasets even
    # though a data/<name>/ dir was staged.
    assert list(SnowDb.open(tmp_path)) == []


def test_open_binds_registered_datasets(tmp_path):
    manager = SnowDbManager.initialize(tmp_path)
    register_dataset_config(manager, 'snodas', config_from_spec(_spec('snodas')))

    assert list(SnowDb.open(tmp_path)) == ['snodas']


def test_open_accepts_the_config_file_directly(tmp_path):
    manager = SnowDbManager.initialize(tmp_path)
    register_dataset_config(manager, 'snodas', config_from_spec(_spec('snodas')))

    opened = SnowDb.open(tmp_path / CONFIG_FILENAME)

    assert opened.root == tmp_path
    assert list(opened) == ['snodas']


def test_open_errors_on_a_dangling_link(tmp_path):
    manager = SnowDbManager.initialize(tmp_path)
    register_dataset_config(manager, 'snodas', config_from_spec(_spec('snodas')))
    # Remove the linked config out of band -> open must fail cleanly.
    (manager.db.data_path / 'snodas' / 'dataset.json').unlink()

    with pytest.raises(SnowDbConfigError, match='missing config'):
        SnowDb.open(tmp_path)


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
    result = manager.rasterize_aois([pourpoint], list(manager.db.registered.values()))

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

    raster = dataset.rasterize_aoi(Pourpoint.from_geojson(pourpoint_geojson))

    assert dataset._aoi_rasters.is_dir()
    assert raster.path.exists()


def test_default_specs_bind_snodas(tmp_path):
    """The built-in DEFAULT_DATASET_SPECS wires up the real snodas spec."""

    db = make_snowdb(tmp_path, DEFAULT_DATASET_SPECS)

    assert db['snodas'].spec.name == 'snodas'


def test_aoi_paths_empty_without_records_dir(tmp_path):
    db = make_snowdb(tmp_path, [_spec('snodas')])

    assert db.pourpoint_paths() == []


def test_aoi_paths_lists_and_sorts_geojson(tmp_path, pourpoint_geojson):
    db = SnowDbManager.initialize(tmp_path).db
    shutil.copy(pourpoint_geojson, db.pourpoint_records_path / 'b.geojson')
    shutil.copy(pourpoint_geojson, db.pourpoint_records_path / 'a.geojson')
    # A non-geojson file is ignored.
    (db.pourpoint_records_path / 'notes.txt').write_text('x')

    assert db.pourpoint_paths() == [
        db.pourpoint_records_path / 'a.geojson',
        db.pourpoint_records_path / 'b.geojson',
    ]


def test_aois_parse_global_geojson(tmp_path, pourpoint_geojson):
    db = SnowDbManager.initialize(tmp_path).db
    shutil.copy(pourpoint_geojson, db.pourpoint_records_path / 'pourpoint.geojson')

    pourpoints = list(db.pourpoints())

    assert len(pourpoints) == 1
    assert pourpoints[0].station_triplet == '12345:MT:USGS'


def test_aoi_triplets(tmp_path, pourpoint_geojson):
    # Triplets are filename-derived (the record filename is authoritative), so
    # the record must be named for its own triplet -- unlike test_aois_parse_
    # global_geojson above, which proves content-based parsing works under an
    # arbitrary filename.
    db = SnowDbManager.initialize(tmp_path).db
    shutil.copy(
        pourpoint_geojson,
        db.pourpoint_records_path / '12345_MT_USGS.geojson',
    )

    assert db.pourpoint_triplets() == {'12345:MT:USGS'}


# --- coverage fallback (3b) --------------------------------------------------


def test_coverage_fallback_none_for_dataset_predating_index(
    tmp_path,
    spec,
    pourpoint_geojson,
):
    # The index is built while only 'test' exists, so its entry carries no key for
    # a dataset registered later. A dataset the index predates reads as NONE (not a
    # raw KeyError), even though geometrically the shared grid would fully cover it.
    manager = make_manager(tmp_path, [spec])
    manager.import_pourpoints(pourpoint_geojson)

    other = make_spec('other', spec, variables=())
    db = make_snowdb(tmp_path, [spec, other])

    assert db.pourpoint_dataset_coverage('12345:MT:USGS', 'other') is Coverage.NONE
    # The dataset the index knows still returns its real coverage.
    assert db.pourpoint_dataset_coverage('12345:MT:USGS', 'test') is Coverage.FULL
    # A genuinely-unknown (unregistered) dataset still raises.
    with pytest.raises(UnknownDatasetError):
        db.pourpoint_dataset_coverage('12345:MT:USGS', 'unregistered')


# --- pourpoint_page (catalog read behind GET /pourpoints) --------------------


def _seed_three_pourpoints(tmp_path, spec):
    """Import three basin-bearing pourpoints with distinct points; return the db.

    Each basin is a small rectangle inside the first tile with its outflow point
    at a distinct longitude, so a point-in-box predicate can pick a subset.
    """
    manager = make_manager(tmp_path, [spec])
    src = tmp_path / 'src'
    src.mkdir()
    # (triplet, point-lon); the basin box is placed around each point's lon.
    for triplet, lon in [
        ('11111:MT:USGS', -119.8),
        ('22222:MT:USGS', -119.5),
        ('33333:MT:USGS', -119.2),
    ]:
        write_pourpoint_record(
            src / f'{triplet.replace(":", "_")}.geojson',
            triplet=triplet,
            box=(lon - 0.05, 44.9, lon + 0.05, 44.0),
            point=(lon, 44.45),
        )
    manager.import_pourpoints(src)
    return manager.db


def test_pourpoint_page_totals_and_pages(tmp_path, spec):
    db = _seed_three_pourpoints(tmp_path, spec)

    # A first page of two: the filtered total is the whole set (3), not the slice.
    page, total = db.pourpoint_page(offset=0, limit=2)
    assert total == 3
    assert [entry.triplet for entry, _ in page] == ['11111:MT:USGS', '22222:MT:USGS']
    # No basins requested -> geometry slot is None (caller uses the point).
    assert all(basin is None for _, basin in page)

    # The second page carries the remainder; the total is unchanged.
    page2, total2 = db.pourpoint_page(offset=2, limit=2)
    assert total2 == 3
    assert [entry.triplet for entry, _ in page2] == ['33333:MT:USGS']


def test_pourpoint_page_filters_on_the_point_predicate(tmp_path, spec):
    db = _seed_three_pourpoints(tmp_path, spec)

    # A predicate selecting only the two westernmost points (lon <= -119.5).
    def contains(lon, lat):
        return lon <= -119.5

    page, total = db.pourpoint_page(offset=0, limit=100, contains=contains)
    assert total == 2  # total is the *filtered* count, before the slice
    assert [entry.triplet for entry, _ in page] == ['11111:MT:USGS', '22222:MT:USGS']


def test_pourpoint_page_loads_basins_when_requested(tmp_path, spec):
    db = _seed_three_pourpoints(tmp_path, spec)

    page, total = db.pourpoint_page(offset=0, limit=1, with_basins=True)
    assert total == 3
    ((entry, basin),) = page
    assert entry.triplet == '11111:MT:USGS'
    # The basin polygon is loaded (the expensive view), not left as None.
    assert basin is not None
    assert basin.type == 'Polygon'


# --- mtime-revalidated index cache (3b) --------------------------------------


def test_index_cache_reloads_after_out_of_band_import(
    tmp_path,
    spec,
    pourpoint_geojson,
):
    # A long-lived reader (e.g. the API's app-scoped SnowDb) picks up an
    # out-of-band import without a restart: the index cache revalidates on mtime.
    manager = SnowDbManager.initialize(tmp_path)
    register_dataset_config(manager, 'test', config_from_spec(spec))

    reader = SnowDb.open(tmp_path)
    assert reader.pourpoint_index().triplets() == set()  # primed empty at open

    SnowDbManager.open(tmp_path).import_pourpoints(pourpoint_geojson)

    assert reader.pourpoint_index().triplets() == {'12345:MT:USGS'}


def test_index_cache_is_stable_then_revalidates_on_mtime(
    tmp_path,
    spec,
    pourpoint_geojson,
):
    manager = SnowDbManager.initialize(tmp_path)
    register_dataset_config(manager, 'test', config_from_spec(spec))
    writer = SnowDbManager.open(tmp_path)
    writer.import_pourpoints(pourpoint_geojson)

    reader = SnowDb.open(tmp_path)
    first = reader.pourpoint_index()
    assert first.triplets() == {'12345:MT:USGS'}
    # No file change -> the identical cached object is returned (one stat, no reload).
    assert reader.pourpoint_index() is first

    # Out-of-band removal rewrites the index; bump the mtime explicitly so the test
    # is independent of the filesystem's mtime resolution.
    writer.remove_pourpoint('12345:MT:USGS')
    idx = reader.pourpoint_index_path
    st = idx.stat()
    os.utime(idx, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    assert reader.pourpoint_index().triplets() == set()


def test_load_basin_raises_on_indexed_point_only_record(
    tmp_path,
    spec,
    pourpoint_geojson,
):
    # The index only lists basin-bearing pourpoints, so an indexed triplet whose
    # on-disk record has been swapped out-of-band to a point-only Feature (no
    # reindex) is a data-integrity bug: `load_basin` raises the typed error
    # rather than returning `None` or the point.
    manager = SnowDbManager.initialize(tmp_path)
    register_dataset_config(manager, 'test', config_from_spec(spec))
    SnowDbManager.open(tmp_path).import_pourpoints(pourpoint_geojson)

    db = SnowDb.open(tmp_path)
    triplet = '12345:MT:USGS'
    assert triplet in db.pourpoint_index()  # still indexed as basin-bearing

    db.pourpoint_record_path(triplet).write_text(
        json.dumps(
            {
                'type': 'Feature',
                'id': triplet,
                'geometry': {'type': 'Point', 'coordinates': [-119.45, 44.45]},
                'properties': {'name': 'Test Basin', 'source': 'test'},
            },
        ),
    )

    with pytest.raises(
        IndexedPourpointMissingBasinError,
        match='has no basin polygon',
    ):
        db.load_basin(triplet)

    # An unindexed triplet still gates on the index first (PourpointNotFound).
    with pytest.raises(PourpointNotFoundError):
        db.load_basin('99999:MT:USGS')


# --- staged dataset registration (3c) ----------------------------------------


def test_staged_dataset_registration_end_to_end(tmp_path, spec, pourpoint_geojson):
    # Import a pourpoint first, then stage + commit a dataset over it. Staging is
    # invisible to readers; the config write is the commit point.
    SnowDbManager.initialize(tmp_path)
    manager = SnowDbManager.open(tmp_path)
    manager.import_pourpoints(pourpoint_geojson)

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


def test_getitem_raises_unknown_dataset_error_for_an_unregistered_name(tmp_path, spec):
    db = make_snowdb(tmp_path, [spec])

    with pytest.raises(
        UnknownDatasetError,
        match=r"No such dataset 'nope'\. Active datasets: test\.",
    ):
        db['nope']


def test_registered_dataset_resolves_active_and_inactive_names(tmp_path, spec):
    # registered_dataset serves the management surface: it resolves anything
    # registered, active or not, unlike __getitem__ (which serves only active).
    manager = SnowDbManager.initialize(tmp_path)
    register_dataset_config(manager, spec.name, config_from_spec(spec), active=False)
    db = SnowDb.open(tmp_path)

    assert spec.name not in db.datasets  # inactive -> __getitem__ would refuse it
    assert db.registered_dataset(spec.name).spec.name == spec.name


def test_registered_dataset_raises_with_the_registered_listing(tmp_path, spec):
    db = make_snowdb(tmp_path, [spec])

    with pytest.raises(
        UnknownDatasetError,
        match=r"No such dataset 'nope'\. Registered datasets: test\.",
    ):
        db.registered_dataset('nope')


def test_registered_dataset_appends_a_caller_hint(tmp_path, spec):
    db = make_snowdb(tmp_path, [spec])

    with pytest.raises(
        UnknownDatasetError,
        match=r'Registered datasets: test\. pass a path',
    ):
        db.registered_dataset('nope', hint=' pass a path')


def test_getitem_raises_unknown_dataset_error_for_an_inactive_name(tmp_path, spec):
    # __getitem__ serves only active datasets: a registered-but-inactive name is
    # unresolvable from this surface too, not just a genuinely-unregistered one --
    # but it gets a pointed "activate it" hint instead of the generic miss, since
    # the fix differs (this benefits every caller: CLI stats, the HTTP API).
    manager = SnowDbManager.initialize(tmp_path)
    register_dataset_config(manager, spec.name, config_from_spec(spec), active=False)

    db = SnowDb.open(tmp_path)

    with pytest.raises(
        UnknownDatasetError,
        match=(
            r"Dataset 'test' is registered but inactive\. "
            r"Activate it with 'snowtool dataset activate test'\."
        ),
    ):
        db['test']


def test_set_dataset_active_rejects_an_unregistered_name(tmp_path):
    manager = SnowDbManager.initialize(tmp_path)

    with pytest.raises(ValueError, match="No registered dataset 'nope'"):
        manager.set_dataset_active('nope', True)


def test_bare_link_without_active_key_reads_as_active(tmp_path, spec):
    # Backward/hand-written-config compatibility: a link JSON with no `active`
    # key round-trips through RootConfig as active=True.
    manager = SnowDbManager.initialize(tmp_path)
    config = config_from_spec(spec)
    ds_dir = manager.db.dataset_dir(spec.name, config)
    ds_dir.mkdir(parents=True, exist_ok=True)
    config.save(ds_dir / DATASET_CONFIG_FILENAME)

    config_path = tmp_path / CONFIG_FILENAME
    raw = json.loads(config_path.read_text())
    raw['datasets'] = {
        spec.name: {'type': 'path', 'path': f'data/{spec.name}/dataset.json'},
    }
    config_path.write_text(json.dumps(raw))

    loaded = RootConfig.load(config_path)
    assert loaded.datasets[spec.name].active is True
    # And the reader serves it: bare links stay live with no config edits.
    assert list(SnowDb.open(tmp_path)) == [spec.name]


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
