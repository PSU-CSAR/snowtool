"""The `dataset` command group, driven against the synthetic snowdb."""

import json

from datetime import date

import pytest

from snowtool.cli import cli
from snowtool.cli._context import CliContext
from snowtool.snowdb.config import CONFIG_FILENAME, RootConfig
from snowtool.snowdb.datasets import DATASET_TEMPLATES, config_from_spec
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.ingest import DateIngest
from snowtool.snowdb.manager import SnowDbManager
from snowtool.snowdb.zones.terrain import terrain_provider

from ..conftest import write_swe_cog
from .conftest import (
    full_marker_out_names,
    full_marker_rasters,
)


def _json(result):
    return json.loads(result.output)


def _row(date, action, source):
    return {'dataset': 'test', 'date': date, 'action': action, 'source': source}


def _generate_zones(runner, cli_obj, source_dem):
    """Generate zone layers for 'test' explicitly (terrain from a local DEM;
    landcover from the root config's declared local source)."""
    return runner.invoke(
        cli,
        ['dataset', 'generate-zones', 'test', '--source', 'terrain', str(source_dem)],
        obj=cli_obj,
    )


def _write_swe(cli_obj, date_str='20180427'):
    write_swe_cog(cli_obj.manager.db['test'], date_str)


def _make_date_dirs(root, name, *date_strs):
    """Create empty ``data/<name>/cogs/<YYYYMMDD>/`` dirs (ingested-date stand-ins)."""
    cogs = root / 'data' / name / 'cogs'
    for date_str in date_strs:
        (cogs / date_str).mkdir(parents=True)


# --- list / info -------------------------------------------------------------


def test_list_reports_dataset_names(runner, cli_obj):
    result = runner.invoke(cli, ['dataset', 'list', '--format', 'json'], obj=cli_obj)

    assert result.exit_code == 0
    assert _json(result) == [{'dataset': 'test', 'active': True}]


def test_info_unknown_dataset_errors(runner, cli_obj):
    result = runner.invoke(cli, ['dataset', 'info', 'nope'], obj=cli_obj)

    assert result.exit_code != 0
    assert 'No such dataset' in result.output


def test_info_after_create(
    runner,
    cli_obj,
    source_dem,
    initialized_root,
    stage_test_dataset,
):
    stage_test_dataset(cli_obj, initialized_root)
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
    assert info['zones']['terrain']['aspect'] is None
    assert info['zones']['terrain']['elevation'] == {'band_step_ft': 1000}
    assert info['is_geographic'] is True
    assert 'swe' in info['variables']
    # Grid details are part of `info`.
    assert info['rows'] == 512
    assert info['n_tiles'] == 4
    # Typed, machine-stable fields: null (not the prose 'varies (geographic)')
    # for a geographic grid's cell area, and two numeric fields (not a prose
    # 'MIN .. MAX' string) for the elevation bracket.
    assert info['cell_area_m2'] is None
    assert info['min_elevation_m'] == -100.0
    assert info['max_elevation_m'] == 4500.0
    assert 'elevation_bracket_m' not in info
    # The table form's prose substitutions ('varies (geographic)', 'MIN .. MAX')
    # are pinned as a renderer case in test_render.py.


# --- dates / values ------------------------------------------------------------


@pytest.mark.parametrize(
    ('args', 'expected_dates'),
    [
        pytest.param([], ['2018-04-27', '2019-01-15'], id='lists-all-ingested-dates'),
        pytest.param(['--start', '20190101'], ['2019-01-15'], id='filters-by-range'),
    ],
)
def test_dates_lists_ingested_dates(
    runner,
    cli_obj,
    initialized_root,
    args,
    expected_dates,
):
    _write_swe(cli_obj, '20180427')
    _write_swe(cli_obj, '20190115')

    result = runner.invoke(
        cli,
        ['dataset', 'dates', 'test', *args, '--format', 'json'],
        obj=cli_obj,
    )

    assert result.exit_code == 0, result.output
    assert _json(result) == [{'date': d} for d in expected_dates]


def test_dates_missing_lists_gap_with_explicit_range(runner, cli_obj, initialized_root):
    # Plumbing only: the domain semantics of missing_dates (gap computation,
    # default-start, inverted-window) are pinned in test_report_diagnostics; here
    # we assert the option wiring renders one real gap as the JSON payload.
    _make_date_dirs(initialized_root, 'test', '20180101', '20180103')

    result = runner.invoke(
        cli,
        [
            'dataset',
            'dates',
            'test',
            '--missing',
            '--start',
            '20180101',
            '--end',
            '20180104',
            '--format',
            'json',
        ],
        obj=cli_obj,
    )

    assert result.exit_code == 0, result.output
    assert _json(result) == [{'date': '2018-01-02'}, {'date': '2018-01-04'}]


def test_dates_missing_no_dates_no_start_errors(runner, cli_obj, initialized_root):
    result = runner.invoke(
        cli,
        ['dataset', 'dates', 'test', '--missing', '--end', '20180101'],
        obj=cli_obj,
    )

    assert result.exit_code != 0
    assert 'no ingested dates' in result.output


def test_values_without_dates_errors(runner, cli_obj):
    result = runner.invoke(cli, ['dataset', 'values', 'test'], obj=cli_obj)

    assert result.exit_code != 0
    assert 'no ingested dates' in result.output


def test_values_with_data(
    runner,
    cli_obj,
    source_dem,
    initialized_root,
    stage_test_dataset,
):
    stage_test_dataset(cli_obj, initialized_root)
    _generate_zones(runner, cli_obj, source_dem)
    _write_swe(cli_obj)

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


def test_staging_then_generate_zones_builds_them(
    runner,
    cli_obj,
    initialized_root,
    source_dem,
    source_nlcd,
    stage_test_dataset,
):
    # Staging (skeleton + AOI rasters) never builds zone layers -- those come
    # only from the explicit generate-zones pass.
    stage_test_dataset(cli_obj, initialized_root)

    data = initialized_root / 'data' / 'test'
    assert not (data / 'terrain' / 'elevation.tif').exists()
    assert not (data / 'landcover' / 'forest_cover_pct.tif').exists()

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


def test_create_requires_template(runner, cli_obj):
    # --template is required: create always stamps a new dataset from a
    # template, never restages an already-registered one.
    result = runner.invoke(cli, ['dataset', 'create', 'test'], obj=cli_obj)

    assert result.exit_code != 0
    assert '--template' in result.output


# --- register (create/register) / activate ------------------------------------


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
    # The unknown-dataset message is now the shared factory's wording (normalized
    # across __getitem__/registered_dataset/set_dataset_active): the "Registered"
    # kind is preserved, still naming the offending token.
    assert "No such dataset 'nope'. Registered datasets:" in result.output


def test_create_unknown_template_errors(runner, tmp_path):
    _, ctx = _empty_ctx(tmp_path)

    result = runner.invoke(
        cli,
        ['dataset', 'create', 'x', '--template', 'nope'],
        obj=ctx,
    )

    # click.Choice rejects an unknown --template at parse time (usage error).
    assert result.exit_code == 2
    assert "Invalid value for '--template'" in result.output


def test_register_registers_an_external_config(runner, cli_obj, initialized_root):

    external = initialized_root / 'staged' / 'dataset.json'
    external.parent.mkdir()
    DATASET_TEMPLATES['snodas'].save(external)

    result = runner.invoke(
        cli,
        ['dataset', 'register', 'snodas', str(external)],
        obj=cli_obj,
    )

    assert result.exit_code == 0, result.output
    assert 'inactive' in result.output
    # 'test' was already registered by the fixture; 'snodas' joins it registered
    # but inactive, so readers keep serving only 'test'.
    db = SnowDb.open(initialized_root)
    assert sorted(db.registered) == ['snodas', 'test']
    assert list(db) == ['test']


def test_register_rejects_an_unusable_config(runner, cli_obj, initialized_root):
    bad = initialized_root / 'bad.json'
    # Right resource, but the grid is missing required fields.
    bad.write_text('{"resource": "snowtool.dataset/v1", "grid": {}, "variables": {}}')

    result = runner.invoke(cli, ['dataset', 'register', 'x', str(bad)], obj=cli_obj)

    assert result.exit_code != 0
    assert 'not a usable dataset config' in result.output.lower()


def test_register_requires_initialized_root(runner, tmp_path, spec):

    cfg = tmp_path / 'd.json'
    config_from_spec(spec).save(cfg)
    obj = CliContext(config=tmp_path)

    result = runner.invoke(cli, ['dataset', 'register', 'test', str(cfg)], obj=obj)

    assert result.exit_code != 0
    assert 'not a snowdb' in result.output


# --- ingest (the dataset-generic seam) ---------------------------------------


def test_ingest_staged_config_path_malformed_is_a_clean_error(
    runner,
    cli_obj,
    source_dem,
    tmp_path,
):
    # A dataset NAME token can be a path to a not-yet-registered (staged) config
    # (resolve_dataset -> _build_staged_dataset); a malformed one must render as
    # a clean one-line SnowDbConfigError, not a raw pydantic traceback.
    staged_dir = tmp_path / 'staged'
    staged_dir.mkdir()
    bad = staged_dir / 'dataset.json'
    bad.write_text('{"resource": "snowtool.dataset/v1", "grid": {}, "variables": {}}')

    result = runner.invoke(
        cli,
        ['dataset', 'ingest', str(bad), str(source_dem)],
        obj=cli_obj,
    )

    assert result.exit_code != 0
    assert 'Traceback' not in result.output
    assert 'not a usable dataset config' in result.output.lower()


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
    runner,
    tmp_path,
    source_dem,
    register_fake_ingester,
):
    # The fake plans two dates, each yielding real marker COGs (one per spec
    # variable) so the run drives the genuine atomic write path and reports both
    # planned dates as ingested. `register_fake_ingester` wires it into INGESTERS.
    def plan(source, dataset):
        for d in (date(2020, 1, 1), date(2020, 1, 2)):
            yield DateIngest(
                date=d,
                source_files=[source],
                out_names=full_marker_out_names(dataset),
                build_rasters=lambda h: full_marker_rasters(dataset, h),
            )

    fake = register_fake_ingester(plan)

    result = runner.invoke(
        cli,
        ['dataset', 'ingest', 'test', str(source_dem), '--format', 'json'],
        obj=CliContext(config=tmp_path),
    )

    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    assert rows == [
        _row('2020-01-01', 'ingested', str(source_dem)),
        _row('2020-01-02', 'ingested', str(source_dem)),
    ]
    assert len(fake.calls) == 1


def test_ingest_converges_and_force_reingests(
    runner,
    tmp_path,
    source_dem,
    register_fake_ingester,
):
    # Converge-by-default: a run whose source hash matches reports "up to date";
    # --force re-ingests. Driven through the real per-date skip/force contract by
    # real marker COGs -- a first ingest establishes the current date, a second
    # unchanged run converges, and --force rebuilds it.
    def plan(source, dataset):
        yield DateIngest(
            date=date(2020, 1, 1),
            source_files=[source],
            out_names=full_marker_out_names(dataset),
            build_rasters=lambda h: full_marker_rasters(dataset, h),
        )

    register_fake_ingester(plan)

    # A first ingest lands the date on disk (source hash stamped) so the next,
    # unchanged run can be recognised as already current.
    first = runner.invoke(
        cli,
        ['dataset', 'ingest', 'test', str(source_dem), '--format', 'json'],
        obj=CliContext(config=tmp_path),
    )
    assert first.exit_code == 0, first.output
    assert json.loads(first.stdout)[0]['action'] == 'ingested'

    converge = runner.invoke(
        cli,
        ['dataset', 'ingest', 'test', str(source_dem), '--format', 'json'],
        obj=CliContext(config=tmp_path),
    )
    assert converge.exit_code == 0, converge.output
    assert json.loads(converge.stdout) == [
        _row('2020-01-01', 'up-to-date', str(source_dem)),
    ]

    forced = runner.invoke(
        cli,
        ['dataset', 'ingest', 'test', '--force', str(source_dem), '--format', 'json'],
        obj=CliContext(config=tmp_path),
    )
    assert forced.exit_code == 0, forced.output
    assert json.loads(forced.stdout) == [
        _row('2020-01-01', 'ingested', str(source_dem)),
    ]


def test_ingest_accepts_a_directory_source(runner, tmp_path, register_fake_ingester):
    # The instarr shape: SOURCE may be a directory (a date's tiles are mosaicked
    # together, so the whole directory is one ingest call).
    def plan(source, dataset):
        # A directory SOURCE fans out to per-date files (the instarr shape);
        # the driver hashes the files, not the directory.
        yield DateIngest(
            date=date(2020, 1, 1),
            source_files=sorted(source.glob('*.nc')),
            out_names=full_marker_out_names(dataset),
            build_rasters=lambda h: full_marker_rasters(dataset, h),
        )

    fake = register_fake_ingester(plan)
    src_dir = tmp_path / 'tiles'
    src_dir.mkdir()
    (src_dir / 'tile.nc').write_bytes(b'tile bytes')

    result = runner.invoke(
        cli,
        ['dataset', 'ingest', 'test', str(src_dir), '--format', 'json'],
        obj=CliContext(config=tmp_path),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == [
        _row('2020-01-01', 'ingested', str(src_dir)),
    ]
    assert fake.calls == [(src_dir, 'test')]


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
    runner,
    tmp_path,
    source_dem,
    register_fake_ingester,
):
    # The register/activate contract end-to-end: a deactivated dataset stays fully
    # manageable by name (ingest), reports active=false, and the read surface
    # (stats) refuses it with a pointed "activate it" error.
    def plan(source, dataset):
        yield DateIngest(
            date=date(2020, 1, 1),
            source_files=[source],
            out_names=full_marker_out_names(dataset),
            build_rasters=lambda h: full_marker_rasters(dataset, h),
        )

    register_fake_ingester(plan)

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
        ['dataset', 'ingest', 'test', str(source_dem), '--format', 'json'],
        obj=ctx(),
    )
    assert ingested.exit_code == 0, ingested.output
    assert json.loads(ingested.stdout) == [
        _row('2020-01-01', 'ingested', str(source_dem)),
    ]

    refused = runner.invoke(
        cli,
        ['stats', 'test', '12345:MT:USGS', '--variable', 'swe'],
        obj=ctx(),
    )
    assert refused.exit_code != 0
    assert 'registered but inactive' in refused.output
    assert 'dataset activate test' in refused.output


# --- generate ----------------------------------------------------------------


@pytest.mark.parametrize(
    ('provider', 'source_fixture', 'message'),
    [
        pytest.param(
            'terrain',
            'source_dem',
            'generated terrain for test',
            id='terrain',
        ),
        pytest.param(
            'landcover',
            'source_nlcd',
            'generated landcover for test',
            id='landcover',
        ),
    ],
)
def test_generate_is_idempotent(
    runner,
    cli_obj,
    request,
    provider,
    source_fixture,
    message,
):
    # Running generate repeatedly overwrites and always succeeds (idempotent),
    # for both the terrain and land-cover providers.
    source = request.getfixturevalue(source_fixture)
    for _ in range(2):
        result = runner.invoke(
            cli,
            [
                'dataset',
                'generate-zones',
                'test',
                '--provider',
                provider,
                '--source',
                provider,
                str(source),
            ],
            obj=cli_obj,
        )
        assert result.exit_code == 0
        assert message in result.output


def test_generate_threads_workers_and_block_size_to_engine(runner, cli_obj, source_dem):
    # --workers and --block-size must reach the terrain engine (the two knobs).
    # Wrap the already-injected fake engine to capture its kwargs and inject the
    # capturing provider through the constructor seam -- reconfiguring the injected
    # dependency, not patching a module global or the private _engine attribute.
    captured = {}
    terrain = next(p for p in cli_obj.zone_layer_providers if p.name == 'terrain')
    inner = terrain.engine

    def _capture(source, targets, bounds, **kwargs):
        captured.update(workers=kwargs.get('workers'), block=kwargs.get('block_size'))
        return inner(source, targets, bounds, **kwargs)

    capturing_cli_obj = CliContext(
        config=cli_obj.config,
        zone_layer_providers=tuple(
            terrain_provider(engine=_capture) if p.name == 'terrain' else p
            for p in cli_obj.zone_layer_providers
        ),
    )

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
        obj=capturing_cli_obj,
    )
    assert result.exit_code == 0
    assert captured['workers'] == 3
    assert captured['block'] == 512


# --- remove-date ---------------------------------------------------------------


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
        ['dataset', 'remove-date', 'test', '2018-01-01', '--yes'],
        obj=cli_obj,
    )

    assert result.exit_code == 0
    assert 'removed test 2018-01-01' in result.output
    assert not (initialized_root / 'data' / 'test' / 'cogs' / '20180101').exists()


def test_remove_absent_date_reports(runner, cli_obj, initialized_root):
    _make_date_dirs(initialized_root, 'test')  # create empty cogs/

    result = runner.invoke(
        cli,
        ['dataset', 'remove-date', 'test', '20180101', '--yes'],
        obj=cli_obj,
    )

    assert result.exit_code == 0
    assert 'absent' in result.output


def test_remove_date_without_yes_refuses(runner, cli_obj, initialized_root):
    # CliRunner's stdin is not a TTY, so this must fail loudly rather than hang
    # or silently proceed -- the same non-interactive case as CI.
    _make_date_dirs(initialized_root, 'test', '20180101')

    result = runner.invoke(
        cli,
        ['dataset', 'remove-date', 'test', '20180101'],
        obj=cli_obj,
    )

    assert result.exit_code != 0
    assert '--yes' in result.output
    assert (initialized_root / 'data' / 'test' / 'cogs' / '20180101').is_dir()


@pytest.mark.parametrize('fmt', ['table', 'json', 'csv'])
def test_list_renders_every_format(runner, cli_obj, fmt):
    result = runner.invoke(cli, ['dataset', 'list', '--format', fmt], obj=cli_obj)

    assert result.exit_code == 0
    assert 'test' in result.output
