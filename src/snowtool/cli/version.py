import click


@click.command()
def version() -> None:
    """Print the snowtool version."""
    from snowtool import __version__

    click.echo(__version__)
