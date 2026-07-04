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
from xml.sax.saxutils import quoteattr

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
                <environmentVariable name="SNOWTOOL_SNOWDB_CONFIG" value=$snowdb_config />
            </environmentVariables>
        </httpPlatform>
    </system.webServer>
</configuration>
""",  # noqa: E501
)


def render_web_config(python_exe: Path, snowdb_config: Path) -> str:
    """Render ``web.config`` content launching ``python_exe`` with ``snowdb_config``.

    ``python_exe`` should be the currently-running interpreter's
    ``sys.executable`` -- the uv-tool venv's python -- so the launched
    process resolves ``snowtool`` without any PATH/shim lookup. Both values
    are filesystem paths rendered as XML attributes, so they're quoted +
    escaped via :func:`xml.sax.saxutils.quoteattr` rather than interpolated
    raw.
    """
    return _WEB_CONFIG_TEMPLATE.substitute(
        python_exe=quoteattr(str(python_exe)),
        snowdb_config=quoteattr(str(snowdb_config)),
    )
