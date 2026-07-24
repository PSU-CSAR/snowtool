"""SnowDb: open/bind, catalog reads, pourpoint paging, and the index cache."""

import json
import os
import shutil

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
    RootConfig,
)
from snowtool.snowdb.coverage import Coverage
from snowtool.snowdb.datasets import DEFAULT_DATASET_SPECS, config_from_spec
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.manager import SnowDbManager
from snowtool.snowdb.spec import DatasetSpec, GridParams

from ..conftest import (
    make_manager,
    make_snowdb,
    make_spec,
    register_dataset_config,
    write_pourpoint_record,
)


def _spec(name: str) -> DatasetSpec:
    """A bare 256x256 single-tile DatasetSpec named ``name`` (no variables/zones).

    Shared by the ``db``/``manager`` tests in this module, which only need a spec
    to bind into a ``SnowDb``/``SnowDbManager`` and don't exercise its variables
    or zones.
    """
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


def test_pourpoints_raises_typed_error_on_basin_less_record(tmp_path, spec):
    # SnowDb.pourpoints() feeds the rasterize/coverage pass and `doctor
    # pourpoints`; every stored record is basin-bearing (the import boundary
    # guarantees it), so a point-only record in records/ is a corrupt store. It
    # must be constructed through Pourpoint.from_basin_record so it raises the
    # typed error naming the file, not the untyped ValueError a downstream
    # `.geometry` access would raise.
    db = SnowDbManager.initialize(tmp_path).db
    triplet = '12345:MT:USGS'
    record = db.pourpoint_records_path / '12345_MT_USGS.geojson'
    write_pourpoint_record(record, triplet, point_only=True)

    with pytest.raises(
        IndexedPourpointMissingBasinError,
        match=r'12345_MT_USGS\.geojson',
    ):
        db.pourpoints()


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


# --- coverage fallback --------------------------------------------------------


def test_coverage_fallback_none_for_dataset_predating_index(
    tmp_path,
    spec,
    pourpoint_geojson,
):
    # The index is built while only 'test' exists, so its entry carries no key for
    # a dataset registered later. A dataset the index predates reads as NONE (not a
    # raw KeyError), even though geometrically the shared grid would fully cover it.
    manager = SnowDbManager.initialize(tmp_path)
    register_dataset_config(manager, 'test', config_from_spec(spec))
    manager = SnowDbManager.open(tmp_path)
    manager.pourpoints.import_(pourpoint_geojson)

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
    SnowDbManager.initialize(tmp_path)
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
    manager.pourpoints.import_(src)
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


# --- mtime-revalidated index cache ---------------------------------------------


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

    SnowDbManager.open(tmp_path).pourpoints.import_(pourpoint_geojson)

    assert reader.pourpoint_index().triplets() == {'12345:MT:USGS'}


def test_index_cache_is_stable_then_revalidates_on_mtime(
    tmp_path,
    spec,
    pourpoint_geojson,
):
    manager = SnowDbManager.initialize(tmp_path)
    register_dataset_config(manager, 'test', config_from_spec(spec))
    writer = SnowDbManager.open(tmp_path)
    writer.pourpoints.import_(pourpoint_geojson)

    reader = SnowDb.open(tmp_path)
    first = reader.pourpoint_index()
    assert first.triplets() == {'12345:MT:USGS'}
    # No file change -> the identical cached object is returned (one stat, no reload).
    assert reader.pourpoint_index() is first

    # Out-of-band removal rewrites the index; bump the mtime explicitly so the test
    # is independent of the filesystem's mtime resolution.
    writer.pourpoints.remove('12345:MT:USGS')
    idx = reader.pourpoint_index_path
    st = idx.stat()
    os.utime(idx, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    assert reader.pourpoint_index().triplets() == set()


def _corrupt_to_point_only(db, triplet):
    """Swap an indexed triplet's on-disk record to a point-only Feature.

    Simulates the corruption case: an out-of-band ``records/`` edit not followed
    by ``pourpoint reindex``, so the index lists the triplet as basin-bearing
    while the stored record has only a point, not a polygon.
    """
    write_pourpoint_record(db.pourpoint_record_path(triplet), triplet, point_only=True)


def test_load_pourpoint_raises_on_indexed_point_only_record(
    tmp_path,
    spec,
    pourpoint_geojson,
):
    # `load_pourpoint` delegates to `Pourpoint.from_basin_record`, the single
    # owner of the `indexed => basin-bearing` invariant: an indexed triplet
    # whose on-disk record has been swapped out-of-band to a point-only
    # Feature (no reindex) is a data-integrity bug, so this test pins that the
    # delegation surfaces the typed error rather than returning a basin-less
    # Pourpoint that downstream code must re-check.
    manager = SnowDbManager.initialize(tmp_path)
    register_dataset_config(manager, 'test', config_from_spec(spec))
    SnowDbManager.open(tmp_path).pourpoints.import_(pourpoint_geojson)

    db = SnowDb.open(tmp_path)
    triplet = '12345:MT:USGS'
    assert triplet in db.pourpoint_index()  # still indexed as basin-bearing
    _corrupt_to_point_only(db, triplet)

    with pytest.raises(
        IndexedPourpointMissingBasinError,
        match='has no basin polygon',
    ):
        db.load_pourpoint(triplet)

    # An unindexed triplet still gates on the index first (PourpointNotFound).
    with pytest.raises(PourpointNotFoundError):
        db.load_pourpoint('99999:MT:USGS')


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


# --- reopened() (the shared fresh-view primitive) ----------------------------


def test_reopened_reflects_a_registration_after_the_snapshot(tmp_path, spec):
    # reopened() re-reads the on-disk root config, so a dataset registered after
    # this SnowDb was opened is visible on the fresh view -- unlike the
    # open-time snapshot, which never refreshes.
    manager = SnowDbManager.initialize(tmp_path)
    snapshot = SnowDb.open(tmp_path)
    assert list(snapshot.registered) == []  # opened before any registration

    register_dataset_config(manager, 'test', config_from_spec(spec))

    assert list(snapshot.registered) == []  # snapshot is frozen
    assert list(snapshot.reopened().registered) == ['test']  # fresh view sees it


def test_reopened_raises_for_a_rootless_in_code_db(tmp_path):
    # A SnowDb built in code with absolute links has no root to re-open, so the
    # shared primitive raises the typed config error rather than a bare one.
    dataset_config = config_from_spec(_spec('snodas'))
    dataset_config.data_dir = tmp_path / 'anywhere' / 'snodas'
    config = RootConfig.create()
    config.pourpoint_records = str(tmp_path / 'pourpoints' / 'records')
    config.pourpoint_index = str(tmp_path / 'pourpoints' / 'index.geojson')
    config.datasets['snodas'] = InlineDatasetLink(dataset=dataset_config)
    db = SnowDb(config)
    assert db.root is None

    with pytest.raises(SnowDbConfigError, match='cannot reopen'):
        db.reopened()


# --- zone_layer_source() (checked generation-source lookup) -------------------


def test_zone_layer_source_raises_when_unconfigured_and_rootless(tmp_path):
    # A rootless in-code db has no default-source fallback (no root to anchor),
    # so a provider whose source was never configured has no entry. The checked
    # lookup raises the typed error naming the fix, not a bare KeyError.
    from snowtool.exceptions import ZoneLayerSourceNotConfiguredError

    dataset_config = config_from_spec(_spec('snodas'))
    dataset_config.data_dir = tmp_path / 'anywhere' / 'snodas'
    config = RootConfig.create()
    config.pourpoint_records = str(tmp_path / 'pourpoints' / 'records')
    config.pourpoint_index = str(tmp_path / 'pourpoints' / 'index.geojson')
    config.datasets['snodas'] = InlineDatasetLink(dataset=dataset_config)
    db = SnowDb(config)
    assert db.root is None
    assert db.zone_layer_sources == {}  # nothing configured, no default fallback

    with pytest.raises(
        ZoneLayerSourceNotConfiguredError,
        match=r'--source terrain PATH',
    ):
        db.zone_layer_source('terrain')


def test_bare_link_without_active_key_reads_as_active(tmp_path, spec):
    # Backward/hand-written-config compatibility: a link JSON with no `active`
    # key round-trips through RootConfig as active=True.
    manager = SnowDbManager.initialize(tmp_path)
    config = config_from_spec(spec)
    ds_dir = config.resolve_data_dir(spec.name, root=manager.db.root)
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
