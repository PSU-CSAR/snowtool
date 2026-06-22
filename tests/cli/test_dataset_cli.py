"""The `dataset` command group, driven against the synthetic snowdb."""

import json

from datetime import date

import pytest

from snowtool.cli import cli
from snowtool.cli._context import CliContext
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.spec import DatasetSpec, GridParams


def _json(result):
    return json.loads(result.output)


def _create(runner, cli_obj, source_dem):
    return runner.invoke(
        cli,
        ['dataset', 'create', 'test', '--dem', str(source_dem)],
        obj=cli_obj,
    )


# --- list / info -------------------------------------------------------------


def test_list_reports_absent_dataset(runner, cli_obj):
    result = runner.invoke(cli, ['dataset', 'list', '--format', 'json'], obj=cli_obj)

    assert result.exit_code == 0
    assert _json(result) == [{'dataset': 'test', 'present': True, 'dates': 0}]


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
    assert info['terrain'] is True
    assert info['landcover'] is True
    assert info['nlcd_hash'] is not None
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
            '--dem',
            str(source_dem),
            '--nlcd',
            str(source_nlcd),
        ],
        obj=cli_obj,
    )

    assert result.exit_code == 0
    assert 'created dataset test' in result.output
    assert 'generated terrain for test' in result.output
    assert 'generated land cover for test' in result.output
    data = initialized_root / 'data' / 'test'
    assert (data / 'terrain' / 'elevation.tif').is_file()
    assert (data / 'terrain' / 'aspect_majority.tif').is_file()
    assert (data / 'terrain' / 'aspect_components.tif').is_file()
    assert (data / 'landcover' / 'forest_cover_pct.tif').is_file()


def test_create_is_idempotent(runner, cli_obj, source_dem):
    args = ['dataset', 'create', 'test', '--dem', str(source_dem)]

    first = runner.invoke(cli, args, obj=cli_obj)
    second = runner.invoke(cli, args, obj=cli_obj)

    assert first.exit_code == 0
    assert 'created dataset test' in first.output
    # The second run is a no-op success, not an error.
    assert second.exit_code == 0
    assert 'already created' in second.output


def test_create_requires_initialized_root(runner, tmp_path, spec, source_dem):
    # An un-initialized root: a write command must refuse rather than create it.
    obj = CliContext(root=tmp_path, specs=(spec,))

    result = runner.invoke(
        cli,
        ['dataset', 'create', 'test', '--dem', str(source_dem)],
        obj=obj,
    )

    assert result.exit_code != 0
    assert 'not an initialized snowdb' in result.output


def test_create_unknown_dataset_errors(runner, cli_obj, source_dem):
    result = runner.invoke(
        cli,
        ['dataset', 'create', 'nope', '--dem', str(source_dem)],
        obj=cli_obj,
    )

    assert result.exit_code != 0
    assert 'No such dataset' in result.output


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


def test_ingest_delegates_to_spec_ingester(runner, tmp_path, spec, source_dem):
    class _FakeIngester:
        def __init__(self):
            self.calls = []

        def ingest(self, source, dataset, *, force=False):
            self.calls.append((source, dataset.spec.name, force))
            return [date(2020, 1, 1), date(2020, 1, 2)]

    ingester = _FakeIngester()
    ingestable = DatasetSpec(
        name='test',
        grid_params=spec.grid_params,
        ingester=ingester,
    )
    SnowDb.initialize(tmp_path, [ingestable])
    obj = CliContext(root=tmp_path, specs=(ingestable,))

    result = runner.invoke(
        cli,
        ['dataset', 'ingest', 'test', str(source_dem)],
        obj=obj,
    )

    assert result.exit_code == 0
    assert 'ingested test 2020-01-01' in result.output
    assert 'ingested test 2020-01-02' in result.output
    assert len(ingester.calls) == 1


# --- generate / rebuild-area -------------------------------------------------


def test_generate_is_idempotent(runner, cli_obj, source_dem):
    _create(runner, cli_obj, source_dem)

    # Running generate repeatedly overwrites and always succeeds (idempotent).
    for _ in range(2):
        result = runner.invoke(
            cli,
            ['dataset', 'generate', 'test', '--source', str(source_dem)],
            obj=cli_obj,
        )
        assert result.exit_code == 0
        assert 'generated terrain for test' in result.output


def test_generate_threads_workers_and_block_size_to_engine(
    runner,
    cli_obj,
    source_dem,
    monkeypatch,
):
    # --workers and --block-size must reach the terrain engine (the two knobs).
    from tests.cli.conftest import _fake_generate_terrain

    captured = {}

    def _capture(source, targets, **kwargs):
        captured.update(workers=kwargs.get('workers'), block=kwargs.get('block_size'))
        return _fake_generate_terrain(source, targets, **kwargs)

    monkeypatch.setattr(
        'snowtool.snowdb.terrain_generate.generate_terrain',
        _capture,
    )
    _create(runner, cli_obj, source_dem)

    result = runner.invoke(
        cli,
        [
            'dataset', 'generate', 'test',
            '--source', str(source_dem),
            '--workers', '3',
            '--block-size', '512',
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
            ['dataset', 'generate-landcover', 'test', '--source', str(source_nlcd)],
            obj=cli_obj,
        )
        assert result.exit_code == 0
        assert 'generated land cover for test' in result.output


def test_rebuild_area_geographic_is_idempotent(runner, cli_obj, source_dem):
    _create(runner, cli_obj, source_dem)

    for _ in range(2):
        result = runner.invoke(cli, ['dataset', 'rebuild-area', 'test'], obj=cli_obj)
        assert result.exit_code == 0
        assert 'rebuilt area raster for test' in result.output


def test_rebuild_area_projected_is_noop(runner, tmp_path):
    spec = DatasetSpec(
        name='utm',
        grid_params=GridParams(
            origin_x=500000.0,
            origin_y=5000000.0,
            px_size=30.0,
            cols=256,
            rows=256,
            tile_size=256,
            crs=32611,
        ),
    )
    SnowDb.initialize(tmp_path, [spec])
    obj = CliContext(root=tmp_path, specs=(spec,))

    result = runner.invoke(cli, ['dataset', 'rebuild-area', 'utm'], obj=obj)

    assert result.exit_code == 0
    assert 'nothing to do' in result.output
    assert not (tmp_path / 'data' / 'utm' / 'areas.tif').exists()


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
