# Windows / IIS

Deploy `snowtool` as an IIS site fronting `snowtool api serve` via
httpPlatformHandler. Install the tool machine-wide first, then provision the
IIS site that runs it.

## Installing snowtool for all users

`uv tool install` and `uv tool update-shell` only ever touch the *installing
user's* profile and PATH — there's no install-time hook to make the tool
available machine-wide. To get `snowtool` onto every user's PATH (including the
IIS app-pool identity that runs the site):

1. Point `uv` at a shared install location instead of its per-user default, in
   an elevated shell, **before installing**:

    ```console
    setx /M UV_TOOL_DIR C:\ProgramData\uv\tools
    setx /M UV_TOOL_BIN_DIR C:\ProgramData\uv\bin
    setx /M UV_PYTHON_INSTALL_DIR C:\ProgramData\uv\python
    ```

    `UV_PYTHON_INSTALL_DIR` matters even though the tool itself lands in
    `UV_TOOL_DIR`: a uv tool venv is a shim around a uv-*managed Python
    interpreter* (see `home` in the venv's `pyvenv.cfg`), which uv otherwise
    downloads into the installing user's profile — where other accounts,
    including the site's app-pool identity, can't read it.

    `setx /M` sets these machine-wide; open a *new* elevated shell so they take
    effect, then install the tool:

    ```console
    uv tool install snowtool --managed-python
    ```

    If `snowtool` was installed before `UV_PYTHON_INSTALL_DIR` was set, `uv
    tool install --reinstall --managed-python snowtool` in a new elevated shell
    rebuilds the venv against a machine-wide interpreter (the stray per-user
    one under `%APPDATA%\uv\python` can then be deleted).

2. Add the shared bin directory to the machine-wide PATH:

    ```console
    snowtool windows add-to-path
    ```

    This must run in an elevated shell. It refuses to proceed (and prints these
    same steps) if it detects a per-user install — e.g. if step 1 was skipped
    and `snowtool` landed under the installing admin's own profile — since
    putting a per-user path on the machine-wide PATH would only work for that
    one account.

    Open a new shell afterward to pick up the change.

## IIS setup

With `snowtool` installed (above), provision the IIS site that hosts the API.

### Prerequisites

On the target Windows Server:

- IIS with the **httpPlatformHandler** module installed, plus the
  **IISAdministration** (version 1.1.0.0+) and **WebAdministration**
  PowerShell modules. Windows Server 2019+ ships both; Server 2016's inbox
  IISAdministration is 1.0.0.0 and must be updated first:
  `Install-Module IISAdministration`.
- An elevated (Administrator) PowerShell/shell to run the install commands.

### Provisioning the site

```console
snowtool windows iis install C:\inetpub\snowtool --hostname snow.example.org --config C:\snowdb\snowdb_conf.json
```

`--config` is written into the site's `web.config` as the
`SNOWTOOL_SNOWDB_CONFIG` environment variable the hosted process reads, and its
directory is granted read+execute to the site's app-pool identity.

Re-running `snowtool windows iis install` against an existing site updates it in
place. Tear a site down with `snowtool windows iis remove` (it also takes
`--config`, used to strip the app-pool identity's permission grant from the
snowdb directory).
