"""The `dataset` command group, driven against the synthetic snowdb."""

import json

from datetime import date

import pytest

from snowtool.cli import cli
from snowtool.cli._context import CliContext
from snowtool.snowdb import datasets as datasets_mod
from snowtool.snowdb.datasets import DATASET_TEMPLATES, config_from_spec
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.ingest import IngestResult
from snowtool.snowdb.manager import SnowDbManager

from ..conftest import register_dataset_config


def _json(result):
    return json.loads(result.output)


def _create(runner, cli_obj, source_dem):
    return runner.invoke(
        cli,
        ['dataset', 'create', 'test', '--source', 'terrain', str(source_dem)],
        obj=cli_obj,
    )


# --- list / info -------------------------------------------------------------


def test_list_reports_dataset_names(runner, cli_obj):
    result = runner.invoke(cli, ['dataset', 'list', '--format', 'json'], obj=cli_obj)

    assert result.exit_code == 0
    assert _json(result) == [{'dataset': 'test'}]


def test_info_unknown_dataset_errors(runner, cli_obj):
    result = runner.invoke(cli, ['dataset', 'info', 'nope'], obj=cli_obj)

    assert result.exit_code != 0
    assert 'No such dataset' in result.output


def test_info_after_create(runner, cli_obj, source_dem):
    _create(runner, cli_obj, source_dem)

    result = runner.invoke(
        cli,
        ['dataset', 'info', 'test', '--format', 'json'],
        obj=cli_obj,
    )

    info = _json(result)
    assert info['name'] == 'test'
    assert info['present'] is True
    assert info['zone_layers']['terrain']['present'] is True
    assert info['zone_layers']['landcover']['present'] is True
    assert info['zone_layers']['landcover']['hash'] is not None
    assert info['is_geographic'] is True
    assert 'swe' in info['variables']


# --- create ------------------------------------------------------------------


def test_create_builds_terrain_and_landcover(
    runner,
    cli_obj,
    initialized_root,
    source_dem,
    source_nlcd,
):
    result = runner.invoke(
        cli,
        [
            'dataset',
            'create',
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

    assert result.exit_code == 0
    assert 'created dataset test' in result.output
    assert 'generated terrain for test' in result.output
    assert 'generated landcover for test' in result.output
    data = initialized_root / 'data' / 'test'
    assert (data / 'terrain' / 'elevation.tif').is_file()
    assert (data / 'terrain' / 'aspect_majority.tif').is_file()
    assert (data / 'terrain' / 'northness.tif').is_file()
    assert (data / 'terrain' / 'eastness.tif').is_file()
    assert (data / 'landcover' / 'forest_cover_pct.tif').is_file()


def test_create_is_idempotent(runner, cli_obj, source_dem):
    args = ['dataset', 'create', 'test', '--source', 'terrain', str(source_dem)]

    first = runner.invoke(cli, args, obj=cli_obj)
    second = runner.invoke(cli, args, obj=cli_obj)

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


def test_create_unknown_dataset_errors(runner, cli_obj, source_dem):
    result = runner.invoke(
        cli,
        ['dataset', 'create', 'nope', '--source', 'terrain', str(source_dem)],
        obj=cli_obj,
    )

    assert result.exit_code != 0
    assert 'No such dataset' in result.output


# --- add / activate (explicit registration) ----------------------------------


def _empty_ctx(tmp_path):
    """A CliContext over a freshly-initialized, dataset-free root."""
    root = tmp_path / 'empty'
    SnowDbManager.initialize(root)
    return root, CliContext(config=root)


def test_create_template_does_not_register_without_activate(runner, tmp_path):
    root, ctx = _empty_ctx(tmp_path)

    result = runner.invoke(
        cli,
        ['dataset', 'create', 'snodas', '--template', 'snodas', '--quick'],
        obj=ctx,
    )

    assert result.exit_code == 0, result.output
    # Staged but not registered: opening from the config serves no datasets.
    assert list(SnowDb.open(root)) == []
    # ... but the staged config is on disk, ready to register.
    assert (root / 'data' / 'snodas' / 'dataset.json').is_file()


def test_create_template_activate_registers(runner, tmp_path):
    root, ctx = _empty_ctx(tmp_path)

    result = runner.invoke(
        cli,
        [
            'dataset',
            'create',
            'snodas',
            '--template',
            'snodas',
            '--quick',
            '--activate',
        ],
        obj=ctx,
    )

    assert result.exit_code == 0, result.output
    assert 'registered snodas' in result.output
    assert list(SnowDb.open(root)) == ['snodas']


def test_create_unknown_template_errors(runner, tmp_path):
    _, ctx = _empty_ctx(tmp_path)

    result = runner.invoke(
        cli,
        ['dataset', 'create', 'x', '--template', 'nope', '--quick'],
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
    # 'test' was already registered by the fixture; 'snodas' joins it.
    assert sorted(SnowDb.open(initialized_root)) == ['snodas', 'test']


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

        def ingest(self, source, dataset, *, force=False):
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

        def ingest(self, source, dataset, *, force=False):
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


# --- generate ----------------------------------------------------------------


def test_generate_is_idempotent(runner, cli_obj, source_dem):
    _create(runner, cli_obj, source_dem)

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
    _create(runner, cli_obj, source_dem)

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


def test_generate_landcover_is_idempotent(runner, cli_obj, source_dem, source_nlcd):
    _create(runner, cli_obj, source_dem)

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
