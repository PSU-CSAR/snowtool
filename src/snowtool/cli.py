from pathlib import Path

import click


@click.group()
def cli() -> None: ...


@cli.command()
def version() -> None:
    from snowtool import __version__

    click.echo(__version__)


@cli.group()
def snowdb() -> None:
    """Snow-database management commands."""


@snowdb.command('init')
@click.argument(
    'path',
    required=False,
    type=click.Path(file_okay=False, path_type=Path),
)
def snowdb_init(path: Path | None) -> None:
    """Create the base snowdb layout at PATH (defaults to the snowdb_path setting).

    Lays out ``aois/``, ``data/``, and a ``data/<dataset>/`` directory for every
    configured dataset. This is the only command that creates the base
    ``aois/``/``data/`` directories, and it is idempotent.
    """
    from snowtool.settings import Settings
    from snowtool.snowdb.datasets import DEFAULT_DATASET_SPECS
    from snowtool.snowdb.db import SnowDb

    root = Settings().snowdb_path if path is None else path
    SnowDb.initialize(root, DEFAULT_DATASET_SPECS)
    click.echo(f'initialized snowdb: {root}')


@cli.group()
def migration() -> None:
    """Data-migration commands."""


@migration.command('aoi-tags')
@click.argument(
    'cog_path',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def migrate_aoi_tags(cog_path: Path) -> None:
    """Rewrite one AOI raster's legacy SNODAS quadkey tags to SNOWTOOL_TILE_BBOX.

    Operates on a single COG, in place, and is idempotent. To migrate a whole
    dataset, drive it with the shell, e.g.:

        find aoi-rasters -name '*.tif' | xargs -n1 snowtool migration aoi-tags
    """
    from snowtool.migration.aoi_tags import migrate_aoi_raster_tags

    try:
        migrated = migrate_aoi_raster_tags(cog_path)
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    click.echo(f'migrated: {cog_path}' if migrated else f'skipped: {cog_path}')


@migration.command('restructure')
@click.argument(
    'src',
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.argument(
    'dst',
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option('--dataset', default='snodas', help='Dataset name under data/.')
def restructure(src: Path, dst: Path, dataset: str) -> None:
    """Move a legacy flat rasterdb dir SRC into snowdb root DST.

    SRC's contents become DST/data/<dataset>/ and DST/aois/ is created.
    """
    from snowtool.migration.restructure import restructure_to_snowdb

    try:
        dataset_dir = restructure_to_snowdb(src, dst, dataset)
    except (ValueError, FileExistsError) as e:
        raise click.ClickException(str(e)) from e

    click.echo(f'restructured: {src} -> {dataset_dir}')
