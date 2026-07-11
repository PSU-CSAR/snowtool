"""The `dataset` command group, driven against the synthetic snowdb."""

import json

from datetime import date

import numpy
import pytest

from snowtool.cli import cli
from snowtool.cli._context import CliContext
from snowtool.snowdb import datasets as datasets_mod
from snowtool.snowdb.config import CONFIG_FILENAME, RootConfig
from snowtool.snowdb.datasets import DATASET_TEMPLATES, config_from_spec
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.ingest import IngestResult
from snowtool.snowdb.manager import SnowDbManager
from snowtool.snowdb.raster.cog import write_cog

from ..conftest import SIZE, SWE_VALUE, TILE, register_dataset_config, snodas_swe_name


def _json(result):
    return json.loads(result.output)


def _create(runner, cli_obj):
    """Stage + register the synthetic dataset (create is stage-only: no zones)."""
    return runner.invoke(cli, ['dataset', 'create', 'test'], obj=cli_obj)


def _generate_zones(runner, cli_obj, source_dem):
    """Generate zone layers for 'test' explicitly (terrain from a local DEM;
    landcover from the root config's declared local source)."""
    return runner.invoke(
        cli,
        ['dataset', 'generate-zones', 'test', '--source', 'terrain', str(source_dem)],
        obj=cli_obj,
    )


def _write_swe(root, grid, date_str='20180427'):
    cogs = root / 'data' / 'test' / 'cogs' / date_str
    cogs.mkdir(parents=True, exist_ok=True)
    write_cog(
        cogs / f'{snodas_swe_name(date_str)}.tif',
        numpy.full((SIZE, SIZE), SWE_VALUE, dtype=numpy.int16),
        transform=grid.base_grid.transform,
        tile_size=TILE,
        predictor=2,
    )


# --- list / info -------------------------------------------------------------


def test_list_reports_dataset_names(runner, cli_obj):
    result = runner.invoke(cli, ['dataset', 'list', '--format', 'json'], obj=cli_obj)

    assert result.exit_code == 0
    assert _json(result) == [{'dataset': 'test', 'active': True}]


def test_info_unknown_dataset_errors(runner, cli_obj):
    result = runner.invoke(cli, ['dataset', 'info', 'nope'], obj=cli_obj)

    assert result.exit_code != 0
    assert 'No such dataset' in result.output


def test_info_after_create(runner, cli_obj, source_dem):
    _create(runner, cli_obj)
    _generate_zones(runner, cli_obj, source_dem)

    result = runner.invoke(
        cli,
        ['dataset', 'info', 'test', '--format', 'json'],
        obj=cli_obj,
    )

    info = _json(result)
    assert info['name'] == 'test'
    assert info['active'] is True
    assert info['present'] is True
    assert info['zone_layers']['terrain']['present'] is True
    assert info['zone_layers']['landcover']['present'] is True
    assert info['zone_layers']['landcover']['hash'] is not None
    assert info['is_geographic'] is True
    assert 'swe' in info['variables']
    # Grid details (formerly `report grid`), now folded into `info`.
    assert info['rows'] == 512
    assert info['n_tiles'] == 4


# --- dates / values ------------------------------------------------------------


def test_dates_lists_ingested_dates(runner, cli_obj, initialized_root, grid):
    _write_swe(initialized_root, grid)

    result = runner.invoke(cli, ['dataset', 'dates', 'test'], obj=cli_obj)

    assert result.exit_code == 0, result.output
    assert '2018-04-27' in result.output


def test_dates_filters_by_range(runner, cli_obj, initialized_root, grid):
    _write_swe(initialized_root, grid)

    result = runner.invoke(
        cli,
        ['dataset', 'dates', 'test', '--start', '20190101'],
        obj=cli_obj,
    )

    assert result.exit_code == 0, result.output
    assert '2018-04-27' not in result.output


def test_dates_gaps_reports_gap(runner, cli_obj, initialized_root):
    cogs = initialized_root / 'data' / 'test' / 'cogs'
    for name in ('20180101', '20180103'):
        (cogs / name).mkdir(parents=True)

    result = runner.invoke(
        cli,
        ['dataset', 'dates', 'test', '--gaps', '--format', 'json'],
        obj=cli_obj,
    )

    row = json.loads(result.output)
    assert row['dates'] == 2
    assert row['gaps'] == 1
    assert row['gap_ranges'] == '2018-01-02..2018-01-02'


def test_values_without_dates_errors(runner, cli_obj):
    result = runner.invoke(cli, ['dataset', 'values', 'test'], obj=cli_obj)

    assert result.exit_code != 0
    assert 'no ingested dates' in result.output


def test_values_with_data(runner, cli_obj, source_dem, initialized_root, grid):
    _create(runner, cli_obj)
    _generate_zones(runner, cli_obj, source_dem)
    _write_swe(initialized_root, grid)

    result = runner.invoke(
        cli,
        ['dataset', 'values', 'test', '--format', 'json'],
        obj=cli_obj,
    )

    rows = json.loads(result.output)
    swe = next(r for r in rows if r['variable'] == 'swe')
    assert swe['min'] == swe['max'] == 50
    assert swe['nodata_pct'] == 0.0


# --- create ------------------------------------------------------------------


def test_create_stages_without_zones_then_generate_zones_builds_them(
    runner,
    cli_obj,
    initialized_root,
    source_dem,
    source_nlcd,
):
    created = _create(runner, cli_obj)

    # Create is stage-only: skeleton + registration, never zone layers.
    assert created.exit_code == 0
    assert 'created dataset test' in created.output
    assert 'generated' not in created.output
    data = initialized_root / 'data' / 'test'
    assert not (data / 'terrain' / 'elevation.tif').exists()
    assert not (data / 'landcover' / 'forest_cover_pct.tif').exists()

    # Zone layers come only from the explicit generate-zones pass.
    generated = runner.invoke(
        cli,
        [
            'dataset',
            'generate-zones',
            'test',
            '--source',
            'terrain',
            str(source_dem),
            '--source',
            'landcover',
            str(source_nlcd),
        ],
        obj=cli_obj,
    )

    assert generated.exit_code == 0
    assert 'generated terrain for test' in generated.output
    assert 'generated landcover for test' in generated.output
    assert (data / 'terrain' / 'elevation.tif').is_file()
    assert (data / 'terrain' / 'aspect_majority.tif').is_file()
    assert (data / 'terrain' / 'northness.tif').is_file()
    assert (data / 'terrain' / 'eastness.tif').is_file()
    assert (data / 'landcover' / 'forest_cover_pct.tif').is_file()


def test_create_is_idempotent(runner, cli_obj):
    first = _create(runner, cli_obj)
    second = _create(runner, cli_obj)

    assert first.exit_code == 0
    assert 'created dataset test' in first.output
    # The second run is a no-op success, not an error.
    assert second.exit_code == 0
    assert 'already created' in second.output


def test_create_requires_initialized_root(runner, tmp_path):
    # An un-initialized root has no config, so opening it (to run any command)
    # fails cleanly rather than silently creating one.
    obj = CliContext(config=tmp_path)

    result = runner.invoke(
        cli,
        ['dataset', 'create', 'test', '--template', 'snodas'],
        obj=obj,
    )

    assert result.exit_code != 0
    assert 'not a snowdb' in result.output


def test_create_unknown_dataset_errors(runner, cli_obj):
    result = runner.invoke(cli, ['dataset', 'create', 'nope'], obj=cli_obj)

    assert result.exit_code != 0
    assert 'No such dataset' in result.output


# --- register (create/add) / activate ----------------------------------------


def _empty_ctx(tmp_path):
    """A CliContext over a freshly-initialized, dataset-free root."""
    root = tmp_path / 'empty'
    SnowDbManager.initialize(root)
    return root, CliContext(config=root)


def test_create_registers_inactive(runner, tmp_path):
    root, ctx = _empty_ctx(tmp_path)

    result = runner.invoke(
        cli,
        ['dataset', 'create', 'snodas', '--template', 'snodas'],
        obj=ctx,
    )

    assert result.exit_code == 0, result.output
    # The registered echo points at the two follow-up steps.
    assert 'registered snodas' in result.output
    assert 'dataset generate-zones snodas' in result.output
    assert 'dataset activate snodas' in result.output
    # Registered inactive: readers see nothing ...
    assert list(SnowDb.open(root)) == []
    # ... but the root config carries the link, flagged inactive, and the staged
    # config is on disk.
    config = RootConfig.load(root / CONFIG_FILENAME)
    assert config.datasets['snodas'].active is False
    assert (root / 'data' / 'snodas' / 'dataset.json').is_file()


def test_create_is_idempotent_and_preserves_the_active_flag(runner, tmp_path):
    # A fresh CliContext per invocation: each real CLI process opens the root
    # config anew, and `create`'s registered-check must see the prior writes.
    root, _ = _empty_ctx(tmp_path)
    args = ['dataset', 'create', 'snodas', '--template', 'snodas']

    first = runner.invoke(cli, args, obj=CliContext(config=root))
    assert first.exit_code == 0, first.output
    activated = runner.invoke(
        cli,
        ['dataset', 'activate', 'snodas'],
        obj=CliContext(config=root),
    )
    assert activated.exit_code == 0, activated.output

    # Re-creating a registered dataset never touches its link: it stays active.
    second = runner.invoke(cli, args, obj=CliContext(config=root))

    assert second.exit_code == 0, second.output
    assert 'registered' not in second.output
    assert RootConfig.load(root / CONFIG_FILENAME).datasets['snodas'].active is True


def test_activate_toggles_reader_visibility(runner, tmp_path):
    root, ctx = _empty_ctx(tmp_path)
    staged = runner.invoke(
        cli,
        ['dataset', 'create', 'snodas', '--template', 'snodas'],
        obj=ctx,
    )
    assert staged.exit_code == 0, staged.output

    activated = runner.invoke(cli, ['dataset', 'activate', 'snodas'], obj=ctx)
    assert activated.exit_code == 0, activated.output
    assert 'activated snodas' in activated.output
    assert list(SnowDb.open(root)) == ['snodas']

    deactivated = runner.invoke(cli, ['dataset', 'deactivate', 'snodas'], obj=ctx)
    assert deactivated.exit_code == 0, deactivated.output
    assert 'deactivated snodas' in deactivated.output
    assert list(SnowDb.open(root)) == []


@pytest.mark.parametrize('command', ['activate', 'deactivate'])
def test_activate_unregistered_name_errors(runner, tmp_path, command):
    _, ctx = _empty_ctx(tmp_path)

    result = runner.invoke(cli, ['dataset', command, 'nope'], obj=ctx)

    assert result.exit_code != 0
    assert "No registered dataset 'nope'" in result.output


def test_create_unknown_template_errors(runner, tmp_path):
    _, ctx = _empty_ctx(tmp_path)

    result = runner.invoke(
        cli,
        ['dataset', 'create', 'x', '--template', 'nope'],
        obj=ctx,
    )

    assert result.exit_code != 0
    assert 'No such template' in result.output


def test_add_registers_an_external_config(runner, cli_obj, initialized_root):

    external = initialized_root / 'staged' / 'dataset.json'
    external.parent.mkdir()
    DATASET_TEMPLATES['snodas'].save(external)

    result = runner.invoke(
        cli,
        ['dataset', 'add', 'snodas', str(external)],
        obj=cli_obj,
    )

    assert result.exit_code == 0, result.output
    assert 'inactive' in result.output
    # 'test' was already registered by the fixture; 'snodas' joins it registered
    # but inactive, so readers keep serving only 'test'.
    db = SnowDb.open(initialized_root)
    assert sorted(db.registered) == ['snodas', 'test']
    assert list(db) == ['test']


def test_add_rejects_an_unusable_config(runner, cli_obj, initialized_root):
    bad = initialized_root / 'bad.json'
    # Right resource, but the grid is missing required fields.
    bad.write_text('{"resource": "snowtool.dataset/v1", "grid": {}, "variables": {}}')

    result = runner.invoke(cli, ['dataset', 'add', 'x', str(bad)], obj=cli_obj)

    assert result.exit_code != 0
    assert 'Not a usable dataset config' in result.output


def test_add_requires_initialized_root(runner, tmp_path, spec):

    cfg = tmp_path / 'd.json'
    config_from_spec(spec).save(cfg)
    obj = CliContext(config=tmp_path)

    result = runner.invoke(cli, ['dataset', 'add', 'test', str(cfg)], obj=obj)

    assert result.exit_code != 0
    assert 'not a snowdb' in result.output


# --- ingest (the dataset-generic seam) ---------------------------------------


def test_ingest_without_ingester_errors(runner, cli_obj, source_dem):
    # The synthetic spec configures no ingester; ingest must fail cleanly.
    result = runner.invoke(
        cli,
        ['dataset', 'ingest', 'test', str(source_dem)],
        obj=cli_obj,
    )

    assert result.exit_code != 0
    assert 'no configured ingester' in result.output


def test_ingest_delegates_to_spec_ingester(
    monkeypatch,
    runner,
    tmp_path,
    spec,
    source_dem,
):

    class _FakeIngester:
        def __init__(self):
            self.calls = []

        def ingest(self, source, dataset, *, force=False, **_):
            self.calls.append((source, dataset.spec.name, force))
            return IngestResult(
                ingested=[date(2020, 1, 1), date(2020, 1, 2)],
                skipped=[],
            )

    fake = _FakeIngester()
    # Ingesters are code, referenced by registry name: register the fake so a
    # config naming it resolves to it.
    monkeypatch.setitem(datasets_mod.INGESTERS, 'fake', fake)

    manager = SnowDbManager.initialize(tmp_path)
    config = config_from_spec(spec)
    config.ingester = 'fake'
    register_dataset_config(manager, 'test', config)

    result = runner.invoke(
        cli,
        ['dataset', 'ingest', 'test', str(source_dem)],
        obj=CliContext(config=tmp_path),
    )

    assert result.exit_code == 0, result.output
    assert 'ingested test 2020-01-01' in result.output
    assert 'ingested test 2020-01-02' in result.output
    assert len(fake.calls) == 1


def test_ingest_converges_and_force_reingests(
    monkeypatch,
    runner,
    tmp_path,
    spec,
    source_dem,
):
    # Converge-by-default: a run whose source hash matches reports "up to date";
    # --force re-ingests. The fake ingester reports skipped vs ingested by force.
    class _ConvergingIngester:
        def __init__(self):
            self.forces = []

        def ingest(self, source, dataset, *, force=False, **_):
            self.forces.append(force)
            d = date(2020, 1, 1)
            if force:
                return IngestResult(ingested=[d], skipped=[])
            return IngestResult(ingested=[], skipped=[d])

    fake = _ConvergingIngester()
    monkeypatch.setitem(datasets_mod.INGESTERS, 'fake', fake)

    manager = SnowDbManager.initialize(tmp_path)
    config = config_from_spec(spec)
    config.ingester = 'fake'
    register_dataset_config(manager, 'test', config)
    converge = runner.invoke(
        cli,
        ['dataset', 'ingest', 'test', str(source_dem)],
        obj=CliContext(config=tmp_path),
    )
    assert converge.exit_code == 0, converge.output
    assert 'up to date test 2020-01-01' in converge.output
    assert 'ingested test' not in converge.output

    forced = runner.invoke(
        cli,
        ['dataset', 'ingest', 'test', '--force', str(source_dem)],
        obj=CliContext(config=tmp_path),
    )
    assert forced.exit_code == 0, forced.output
    assert 'ingested test 2020-01-01' in forced.output
    assert 'up to date' not in forced.output
    assert fake.forces == [False, True]


def test_ingest_accepts_a_directory_source(monkeypatch, runner, tmp_path, spec):
    # The instarr shape: SOURCE may be a directory (a date's tiles are mosaicked
    # together, so the whole directory is one ingest call).
    class _FakeIngester:
        def __init__(self):
            self.sources = []

        def ingest(self, source, dataset, *, force=False, **_):
            self.sources.append(source)
            return IngestResult(ingested=[date(2020, 1, 1)], skipped=[])

    fake = _FakeIngester()
    monkeypatch.setitem(datasets_mod.INGESTERS, 'fake', fake)

    manager = SnowDbManager.initialize(tmp_path)
    config = config_from_spec(spec)
    config.ingester = 'fake'
    register_dataset_config(manager, 'test', config)
    src_dir = tmp_path / 'tiles'
    src_dir.mkdir()

    result = runner.invoke(
        cli,
        ['dataset', 'ingest', 'test', str(src_dir)],
        obj=CliContext(config=tmp_path),
    )

    assert result.exit_code == 0, result.output
    assert 'ingested test 2020-01-01' in result.output
    assert fake.sources == [src_dir]


def test_ingest_takes_exactly_one_source(runner, cli_obj, source_dem, source_nlcd):
    # Batch driving belongs to the shell (xargs); a second SOURCE is a usage error.
    result = runner.invoke(
        cli,
        ['dataset', 'ingest', 'test', str(source_dem), str(source_nlcd)],
        obj=cli_obj,
    )

    assert result.exit_code != 0
    assert 'unexpected extra argument' in result.output.lower()


def test_inactive_dataset_is_manageable_but_not_queryable(
    monkeypatch,
    runner,
    tmp_path,
    spec,
    source_dem,
):
    # The register/activate contract end-to-end: a deactivated dataset stays fully
    # manageable by name (ingest), reports active=false, and the read surface
    # (stats) refuses it with a pointed "activate it" error.
    class _FakeIngester:
        def ingest(self, source, dataset, *, force=False, **_):
            return IngestResult(ingested=[date(2020, 1, 1)], skipped=[])

    monkeypatch.setitem(datasets_mod.INGESTERS, 'fake', _FakeIngester())

    manager = SnowDbManager.initialize(tmp_path)
    config = config_from_spec(spec)
    config.ingester = 'fake'
    register_dataset_config(manager, 'test', config)

    # A fresh CliContext per invocation: each real CLI process opens the root
    # config anew, so post-deactivate commands must see the flipped flag.
    def ctx():
        return CliContext(config=tmp_path)

    deactivated = runner.invoke(cli, ['dataset', 'deactivate', 'test'], obj=ctx())
    assert deactivated.exit_code == 0, deactivated.output

    listed = runner.invoke(cli, ['dataset', 'list', '--format', 'json'], obj=ctx())
    assert _json(listed) == [{'dataset': 'test', 'active': False}]

    info = runner.invoke(
        cli,
        ['dataset', 'info', 'test', '--format', 'json'],
        obj=ctx(),
    )
    assert info.exit_code == 0, info.output
    assert _json(info)['active'] is False

    ingested = runner.invoke(
        cli,
        ['dataset', 'ingest', 'test', str(source_dem)],
        obj=ctx(),
    )
    assert ingested.exit_code == 0, ingested.output
    assert 'ingested test 2020-01-01' in ingested.output

    refused = runner.invoke(cli, ['stats', 'test', '12345:MT:USGS'], obj=ctx())
    assert refused.exit_code != 0
    assert 'registered but inactive' in refused.output
    assert 'dataset activate test' in refused.output


# --- generate ----------------------------------------------------------------


def test_generate_is_idempotent(runner, cli_obj, source_dem):
    _create(runner, cli_obj)

    # Running generate repeatedly overwrites and always succeeds (idempotent).
    for _ in range(2):
        result = runner.invoke(
            cli,
            [
                'dataset',
                'generate-zones',
                'test',
                '--provider',
                'terrain',
                '--source',
                'terrain',
                str(source_dem),
            ],
            obj=cli_obj,
        )
        assert result.exit_code == 0
        assert 'generated terrain for test' in result.output


def test_generate_threads_workers_and_block_size_to_engine(runner, cli_obj, source_dem):
    # --workers and --block-size must reach the terrain engine (the two knobs).
    # Wrap the already-injected engine to capture its kwargs -- reconfiguring the
    # injected dependency, not patching a module global.
    captured = {}
    terrain = next(p for p in cli_obj.zone_layer_providers if p.name == 'terrain')
    inner = terrain._engine

    def _capture(source, targets, **kwargs):
        captured.update(workers=kwargs.get('workers'), block=kwargs.get('block_size'))
        return inner(source, targets, **kwargs)

    terrain._engine = _capture
    _create(runner, cli_obj)

    result = runner.invoke(
        cli,
        [
            'dataset',
            'generate-zones',
            'test',
            '--provider',
            'terrain',
            '--source',
            'terrain',
            str(source_dem),
            '--workers',
            '3',
            '--block-size',
            '512',
        ],
        obj=cli_obj,
    )
    assert result.exit_code == 0
    assert captured['workers'] == 3
    assert captured['block'] == 512


def test_generate_landcover_is_idempotent(runner, cli_obj, source_nlcd):
    _create(runner, cli_obj)

    for _ in range(2):
        result = runner.invoke(
            cli,
            [
                'dataset',
                'generate-zones',
                'test',
                '--provider',
                'landcover',
                '--source',
                'landcover',
                str(source_nlcd),
            ],
            obj=cli_obj,
        )
        assert result.exit_code == 0
        assert 'generated landcover for test' in result.output


# --- remove-date / prune -----------------------------------------------------


def _make_date_dirs(root, name, *date_strs):
    cogs = root / 'data' / name / 'cogs'
    for date_str in date_strs:
        (cogs / date_str).mkdir(parents=True)


def test_remove_date_dry_run_keeps_data(runner, cli_obj, initialized_root):
    _make_date_dirs(initialized_root, 'test', '20180101')

    result = runner.invoke(
        cli,
        ['dataset', 'remove-date', 'test', '20180101', '--dry-run'],
        obj=cli_obj,
    )

    assert result.exit_code == 0
    assert 'would remove test 2018-01-01' in result.output
    assert (initialized_root / 'data' / 'test' / 'cogs' / '20180101').is_dir()


def test_remove_date_deletes(runner, cli_obj, initialized_root):
    _make_date_dirs(initialized_root, 'test', '20180101')

    result = runner.invoke(
        cli,
        ['dataset', 'remove-date', 'test', '2018-01-01'],
        obj=cli_obj,
    )

    assert result.exit_code == 0
    assert 'removed test 2018-01-01' in result.output
    assert not (initialized_root / 'data' / 'test' / 'cogs' / '20180101').exists()


def test_remove_absent_date_reports(runner, cli_obj, initialized_root):
    _make_date_dirs(initialized_root, 'test')  # create empty cogs/

    result = runner.invoke(
        cli,
        ['dataset', 'remove-date', 'test', '20180101'],
        obj=cli_obj,
    )

    assert result.exit_code == 0
    assert 'absent' in result.output


def test_prune_removes_only_older_dates(runner, cli_obj, initialized_root):
    _make_date_dirs(initialized_root, 'test', '20180101', '20180201', '20180301')

    result = runner.invoke(
        cli,
        ['dataset', 'prune', 'test', '--before', '20180201'],
        obj=cli_obj,
    )

    assert result.exit_code == 0
    assert 'removed test 2018-01-01' in result.output
    cogs = initialized_root / 'data' / 'test' / 'cogs'
    assert not (cogs / '20180101').exists()
    assert (cogs / '20180201').is_dir()
    assert (cogs / '20180301').is_dir()


def test_prune_dry_run_keeps_data(runner, cli_obj, initialized_root):
    _make_date_dirs(initialized_root, 'test', '20180101')

    result = runner.invoke(
        cli,
        ['dataset', 'prune', 'test', '--before', '20180201', '--dry-run'],
        obj=cli_obj,
    )

    assert 'would remove test 2018-01-01' in result.output
    assert (initialized_root / 'data' / 'test' / 'cogs' / '20180101').is_dir()


def test_prune_nothing_to_do(runner, cli_obj, initialized_root):
    _make_date_dirs(initialized_root, 'test', '20180301')

    result = runner.invoke(
        cli,
        ['dataset', 'prune', 'test', '--before', '20180101'],
        obj=cli_obj,
    )

    assert result.exit_code == 0
    assert 'no dates before' in result.output


@pytest.mark.parametrize('fmt', ['table', 'json', 'csv'])
def test_list_renders_every_format(runner, cli_obj, fmt):
    result = runner.invoke(cli, ['dataset', 'list', '--format', fmt], obj=cli_obj)

    assert result.exit_code == 0
    assert 'test' in result.output
