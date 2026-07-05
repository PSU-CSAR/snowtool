# Contributing

## Project Setup

This project uses [uv](https://docs.astral.sh/uv) for various project
management tasks. This can be installed globally with `brew install uv` on mac.

### Virtual Env

`uv` will manage its own venv.

### Install Project Requirements

Sync the project requirements, with all dev dependencies:

```commandline
uv sync --dev
```

## Git hooks (prek)

Git hooks are run by [prek](https://prek.j178.dev), a pre-commit-compatible
runner (same `.pre-commit-config.yaml`). All hooks are `language: system`:
every tool (ruff, mypy, the file-hygiene checks) runs from the project venv at
its `uv.lock`-pinned version -- prek never builds isolated environments or
installs anything itself. Enable the hooks with:

```commandline
uv run prek install
```

The hooks run against staged content during a `git commit` (unstaged changes
are stashed and restored around the run, so partially staged files are handled
correctly), or run them explicitly against everything with:

```commandline
uv run prek run --all-files
```

If for some reason, you wish to commit code that does not pass the
checks, this can be done with:

```commandline
git commit -m "message" --no-verify
```

## Testing

Tests are run using `pytest`. Put `pytest` python modules and other resource in
the `tests/` directory.

Source the .env file. The app uses Pydantic BaseSettings, but has dotenv
support disabled, so the environment variables must be set.

```commandline
. .env
```

Run the tests:

```commandline
pytest
```

## Documentation

Docs are built with [MkDocs](https://www.mkdocs.org) (Material theme) from the
`docs/` directory and `mkdocs.yml`. The docs dependencies live in the `docs`
dependency group, so run MkDocs through `uv run --group docs` (uv installs the
group on first use — no separate `uv sync` step needed):

```commandline
uv run --group docs mkdocs serve
```

This serves a live-reloading preview at `http://localhost:8000`. Build a static
site into `site/` instead with:

```commandline
uv run --group docs mkdocs build --strict
```

`--strict` fails the build on broken links or warnings (matching CI). The CLI
and Python API reference pages are generated from the source (mkdocs-click and
mkdocstrings), and the HTTP API page is rendered from the tested OpenAPI golden
snapshot via `docs/hooks.py`, so no live snowdb is required.

## Modifying Dependencies

With `uv`, adding dependencies is as simple as running `uv add`. Dev
dependencies can be added by specifying the extra flag `--dev`. Upgrade
dependencies by running `uv sync --upgrade` and optionally passing a package
name to upgrade. By default that command will upgrade all dependencies.

## Dev Server

Run the app with uvicorn:

```commandline
uvicorn snowtool.api.app:get_app --factory --reload
```

With the `uvicorn` defaults, the app should be accessible at
`http://localhost:8000`.

For deploying the tool (including Windows/IIS), see the README.
