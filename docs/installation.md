# Installation

`snowtool` requires **Python 3.14+**.

## As a uv tool

Install it as a standalone [uv](https://docs.astral.sh/uv) tool:

```console
uv tool install snowtool
```

This puts the `snowtool` command on your PATH. To make it available to *all*
users on a Windows Server (rather than just the installing account), see
[Deployment → Windows / IIS](deployment/windows-iis.md#making-snowtool-available-to-all-users).

## From source

For development, clone the repository and sync with uv:

```console
git clone https://github.com/PSU-CSAR/django-snow
cd django-snow
uv sync --dev
```

See [`CONTRIBUTING.md`](https://github.com/PSU-CSAR/django-snow/blob/main/CONTRIBUTING.md)
for the full development setup (pre-commit, testing, and conventions).
