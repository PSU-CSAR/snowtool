from pathlib import Path

import click


@click.group()
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


@migration.command('stamp')
@click.argument(
    'root',
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def stamp(root: Path) -> None:
    """Write a root config into a legacy snowdb ROOT that lacks one.

    A snowdb built before the root config existed has no ``snowdb_conf.json``;
    ``snowtool snowdb`` now requires it. This stamps a default config (no datasets
    registered) into ROOT. Idempotent -- an existing config is left untouched.
    """
    from snowtool.migration.stamp import stamp_root

    try:
        config_path, written = stamp_root(root)
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    if written:
        click.echo(f'stamped: {config_path}')
    else:
        click.echo(f'skipped (already stamped): {config_path}')


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

    SRC's contents become DST/data/<dataset>/ and DST/pourpoints/ is created.
    """
    from snowtool.migration.restructure import restructure_to_snowdb

    try:
        dataset_dir = restructure_to_snowdb(src, dst, dataset)
    except (ValueError, FileExistsError) as e:
        raise click.ClickException(str(e)) from e

    click.echo(f'restructured: {src} -> {dataset_dir}')
