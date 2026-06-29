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

## Pre-commit

Run pre-commit install to enable the pre-commit configuration:

```commandline
pre-commit install
```

The pre-commit hooks will be run against all files during a `git commit`, or
you can run it explicitly with:

```commandline
pre-commit run --all-files
```

If for some reason, you wish to commit code that does not pass the
pre-commit checks, this can be done with:

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
