"""Versioned provenance tags: a content/geometry digest plus the on-disk *format
version* of the artifact it guards.

Every provenance/identity hash stamped on a raster is stored as
``v{version}:{digest}``. The version is the format of the artifact the hash
guards, owned by whatever produces that artifact (a zone-layer provider's
``format_version``; the AOI-raster writer's ``AOI_RASTER_FORMAT_VERSION``);
the digest covers the content/geometry. Storing them together means a material
change to the on-disk format is caught by the very same equality check that
already catches a content change: bump the producer's format version and every
artifact written under the old version reads as stale (forcing a rebuild), even
though its underlying digest is unchanged.

All versions start at 1: this is a greenfield database, so there is no old data
to stay backward-compatible with.
"""

from __future__ import annotations


def versioned_hash(version: int, digest: str) -> str:
    """A provenance digest prefixed with the on-disk format version it guards."""
    return f'v{version}:{digest}'


def parse_format_version(versioned: str | None) -> int | None:
    """The format version encoded by :func:`versioned_hash`, else ``None``.

    Returns ``None`` for a missing value or one not in ``v{int}:{digest}`` form
    (an untagged/legacy hash) -- the caller decides whether that is a finding.
    """
    if not versioned or not versioned.startswith('v'):
        return None
    prefix, sep, _ = versioned.partition(':')
    if not sep:
        return None
    try:
        return int(prefix[1:])
    except ValueError:
        return None
