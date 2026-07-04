# Configuration

`snowtool` reads its snowdb location from the `SNOWTOOL_SNOWDB_CONFIG`
environment variable, or the per-command `--config`/`-C` option.

Dotenv loading is **disabled**, so the variable must be set in the actual
environment — a `.env` file is not read automatically:

```console
export SNOWTOOL_SNOWDB_CONFIG=/path/to/snowdb/snowdb_conf.json
```

The value may point at either the root config file (`snowdb_conf.json`) or the
snowdb directory that contains it.

Commands that don't open a database (`snowtool --version`, `snowtool api serve`
with its own config, the `snowtool windows` group) don't require it.
