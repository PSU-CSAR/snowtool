"""Renders the IIS ``web.config`` that fronts ``snowtool api serve``.

httpPlatformHandler launches a child process and reverse-proxies to it over
loopback; ``processPath``/``arguments`` here launch the *same* interpreter
this command runs under (the uv-tool venv's ``python.exe``, passed in as
``python_exe``) via ``python -m snowtool api serve``, so IIS reuses the
existing CLI command's settings/``--check`` preflight instead of a
hand-rolled ``uvicorn`` invocation. ``SNOWTOOL_SNOWDB_CONFIG`` is passed
through as an environment variable, matching how every other snowtool
entrypoint reads it.

Built with :class:`string.Template` (``$name`` placeholders) rather than
``str.format``/f-strings: IIS's own rewrite-rule macros (``{HTTP_HOST}``,
``{SERVER_PORT_SECURE}``, ...) use curly-brace syntax, which would collide
with ``{}``-style interpolation and need escaping throughout the template.
"""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import TYPE_CHECKING
from xml.sax.saxutils import quoteattr

if TYPE_CHECKING:
    from collections.abc import Mapping

_WEB_CONFIG_TEMPLATE = Template(
    r"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <system.webServer>
        <rewrite>
            <rules>
                <rule name="HTTP to HTTPS Redirect" enabled="true" stopProcessing="true">
                    <match url="(.*)" />
                    <conditions logicalGrouping="MatchAny">
                        <add input="{SERVER_PORT_SECURE}" pattern="^0$$" />
                    </conditions>
                    <action type="Redirect" url="https://{HTTP_HOST}{REQUEST_URI}" redirectType="Permanent" />
                </rule>
            </rules>
        </rewrite>
        <handlers>
            <add name="httpPlatformHandler" path="*" verb="*" modules="httpPlatformHandler" resourceType="Unspecified" requireAccess="Script" />
        </handlers>

        <httpPlatform processPath=$python_exe arguments="-m snowtool api serve --port %HTTP_PLATFORM_PORT%" startupTimeLimit="120" startupRetryCount="3" requestTimeout="00:04:00" stdoutLogEnabled="true" stdoutLogFile=".\log\httpplatform-stdout">
            <environmentVariables>
                <environmentVariable name="SNOWTOOL_SNOWDB_CONFIG" value=$snowdb_config />$data_env_vars
            </environmentVariables>
        </httpPlatform>
    </system.webServer>
</configuration>
""",  # noqa: E501
)


def rasterio_data_env() -> dict[str, Path]:
    """Env vars pinning GDAL/PROJ data to the rasterio wheel's bundled copies.

    Shared GIS servers commonly carry ambient ``GDAL_DATA``/``PROJ_LIB``/
    ``PROJ_DATA`` pointing at some other installation's -- PostGIS, ArcGIS,
    QGIS -- data files, which the hosted process would inherit through IIS
    and which can be arbitrarily incompatible with the wheel's own GDAL/PROJ
    (down to import-time ``proj.db`` layout errors). Pinning all three
    spellings (``PROJ_DATA`` is PROJ >= 9.1's name, ``PROJ_LIB`` the legacy
    one) to the wheel's bundled data makes the site immune. pyproj honors
    these too, so it reads rasterio's PROJ database as well -- the price of
    the immunity.
    """
    import rasterio  # heavy (loads GDAL), deferred to render time

    package = Path(rasterio.__file__).parent
    dirs = {
        'GDAL_DATA': package / 'gdal_data',
        'PROJ_DATA': package / 'proj_data',
        'PROJ_LIB': package / 'proj_data',
    }
    return {name: path for name, path in dirs.items() if path.is_dir()}


def render_web_config(
    python_exe: Path,
    snowdb_config: Path,
    data_env: Mapping[str, Path] | None = None,
) -> str:
    """Render ``web.config`` content launching ``python_exe`` with ``snowdb_config``.

    ``python_exe`` should be the currently-running interpreter's
    ``sys.executable`` -- the uv-tool venv's python -- so the launched
    process resolves ``snowtool`` without any PATH/shim lookup. ``data_env``
    entries (:func:`rasterio_data_env`) become additional
    ``<environmentVariable>`` elements. All values are filesystem paths
    rendered as XML attributes, so they're quoted + escaped via
    :func:`xml.sax.saxutils.quoteattr` rather than interpolated raw.
    """
    data_env_vars = ''.join(
        f'\n                <environmentVariable name="{name}" '
        f'value={quoteattr(str(path))} />'
        for name, path in (data_env or {}).items()
    )
    return _WEB_CONFIG_TEMPLATE.substitute(
        python_exe=quoteattr(str(python_exe)),
        snowdb_config=quoteattr(str(snowdb_config)),
        data_env_vars=data_env_vars,
    )
