# Installation

`snowtool` requires **Python 3.14+**.

## As a uv tool

Install it as a standalone [uv](https://docs.astral.sh/uv) tool:

```console
uv tool install snowtool
```

This puts the `snowtool` command on your PATH. To make it available to *all*
users on a Windows Server (rather than just the installing account), see
[Deployment → Windows / IIS](deployment/windows-iis.md#installing-snowtool-for-all-users).

## From source

For development, clone the repository and sync with uv:

```console
git clone https://github.com/PSU-CSAR/snowtool
cd snowtool
uv sync --dev
```

See [`CONTRIBUTING.md`](https://github.com/PSU-CSAR/snowtool/blob/main/CONTRIBUTING.md)
for the full development setup (git hooks, testing, and conventions).
