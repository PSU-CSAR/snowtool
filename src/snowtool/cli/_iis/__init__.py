"""Implementation behind ``snowtool iis`` (IIS site install/remove).

Kept out of ``snowtool.cli.iis`` so that module stays a thin click shell: the
render/argv-building functions *and* the install/remove orchestration
(:func:`~snowtool.cli._iis.provisioning.install_site` /
:func:`~snowtool.cli._iis.provisioning.remove_site`) live here, importable and
unit-testable without Windows, IIS, or a real ``powershell.exe`` -- the two
I/O seams (``echo``, ``runner``) are injectable.
"""
