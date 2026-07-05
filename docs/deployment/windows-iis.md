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
    ```

    `setx /M` sets these machine-wide; open a *new* elevated shell so they take
    effect, then install the tool:

    ```console
    uv tool install snowtool
    ```

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

- IIS with the **httpPlatformHandler** and **IISAdministration** modules
  installed.
- An elevated (Administrator) PowerShell/shell to run the install commands.

### Provisioning the site

```console
snowtool windows iis install C:\inetpub\snowtool --hostname snow.example.org --config C:\snowdb\snowdb_conf.json
```

`--config` is written into the site's `web.config` as the
`SNOWTOOL_SNOWDB_CONFIG` environment variable the hosted process reads, and its
directory is granted read+execute to the site's app-pool identity.

Re-running `snowtool windows iis install` against an existing site updates it in
place. Tear a site down with `snowtool windows iis remove`.
