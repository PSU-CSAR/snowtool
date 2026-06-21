"""CLI test helpers: a runner plus an initialized synthetic snowdb context.

Commands run against the small synthetic `spec` (top-level conftest), injected
into the CLI via a pre-seeded CliContext on ctx.obj -- the root `cli` group
honors an existing context, so `runner.invoke(cli, args, obj=cli_obj)` drives the
real commands against the tiny grid instead of the full snodas spec.
"""

import pytest

from click.testing import CliRunner

from snowtool.cli._context import CliContext
from snowtool.snowdb.db import SnowDb


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def initialized_root(tmp_path, spec):
    """An initialized snowdb root bound to the synthetic spec."""
    SnowDb.initialize(tmp_path, [spec])
    return tmp_path


@pytest.fixture
def cli_obj(initialized_root, spec) -> CliContext:
    """A CliContext over the initialized synthetic snowdb (inject as obj=)."""
    return CliContext(root=initialized_root, specs=(spec,))
