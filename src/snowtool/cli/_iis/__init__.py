"""Implementation behind ``snowtool iis`` (IIS site install/remove).

Kept out of ``snowtool.cli.iis`` so that module stays a thin click shell: pure
render/argv-building functions live here, importable and unit-testable without
Windows, IIS, or a real ``powershell.exe``.
"""
