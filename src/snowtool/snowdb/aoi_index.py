"""The derived AOI manifest: a GeoJSON ``FeatureCollection`` index of the records.

The per-AOI geojson under ``aois/records/`` are the lossless source of truth, but
parsing every one (each basin is thousands of coordinate pairs) just to list the
AOIs is wasteful. :class:`AOIIndex` is a lightweight, rebuildable index of the
list-relevant fields -- one ``Point`` ``Feature`` per AOI (``id`` = triplet,
``geometry`` = the pourpoint, ``properties`` = name/source/active/basinarea + the
basin :attr:`~snowtool.snowdb.aoi.AOI.geometry_hash`) -- persisted to
``aois/index.geojson``. It is GeoJSON-native on purpose: the same file is a
plottable point layer and the FastAPI listing payload. Being derived, it is always
rebuildable from ``records/`` (``aoi reindex``), so it never has to be trusted as
primary data.
"""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from snowtool import types
from snowtool.snowdb.aoi import AOI
from snowtool.snowdb.coverage import Coverage, dataset_coverage

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from snowtool.snowdb.coverage import CoverageDomain


@dataclass(frozen=True)
class AOIIndexEntry:
    """One AOI's denormalized list-fields (a single manifest ``Feature``)."""

    triplet: types.StationTriplet
    name: str
    source: str
    point: dict[str, Any]
    geometry_hash: str
    active: bool | None = None
    basinarea: float | None = None
    # Per-dataset geometric coverage of this AOI's basin, keyed by dataset name.
    # Derived (a grid change => `aoi reindex`), so it is recomputed by
    # `AOIIndex.from_records`, never edited in place; absent for a point-only AOI
    # (which is not indexed anyway).
    coverage: dict[str, Coverage] = field(default_factory=dict)

    @classmethod
    def from_aoi(
        cls: type[Self],
        aoi: AOI,
        domains: Mapping[str, CoverageDomain],
    ) -> Self:
        return cls(
            triplet=aoi.station_triplet,
            name=aoi.name,
            source=aoi.source,
            point=aoi.point,
            geometry_hash=aoi.geometry_hash,
            active=aoi.properties.get('active'),
            basinarea=aoi.properties.get('basinarea'),
            coverage={
                name: dataset_coverage(aoi, domain)
                for name, domain in domains.items()
            },
        )

    def to_feature(self: Self) -> dict[str, Any]:
        return {
            'type': 'Feature',
            'id': self.triplet,
            'geometry': self.point,
            'properties': {
                'name': self.name,
                'source': self.source,
                'active': self.active,
                'basinarea': self.basinarea,
                'geometry_hash': self.geometry_hash,
                'coverage': {
                    name: cov.value for name, cov in self.coverage.items()
                },
            },
        }

    @classmethod
    def from_feature(cls: type[Self], feature: dict[str, Any]) -> Self:
        properties = feature['properties']
        return cls(
            triplet=types.StationTriplet(feature['id']),
            name=properties['name'],
            source=properties['source'],
            point=feature['geometry'],
            geometry_hash=properties['geometry_hash'],
            active=properties.get('active'),
            basinarea=properties.get('basinarea'),
            coverage={
                name: Coverage(value)
                for name, value in properties['coverage'].items()
            },
        )


@dataclass
class AOIIndex:
    """The set of :class:`AOIIndexEntry`, keyed by triplet.

    Serializes to / from a GeoJSON ``FeatureCollection`` (features sorted by
    triplet for stable, reviewable diffs). Build it from the ``records/`` dir
    (the rebuild path) or load it from a persisted ``index.geojson``.
    """

    entries: dict[types.StationTriplet, AOIIndexEntry]

    @classmethod
    def from_entries(cls: type[Self], entries: Iterable[AOIIndexEntry]) -> Self:
        return cls({entry.triplet: entry for entry in entries})

    @classmethod
    def from_records(
        cls: type[Self],
        records_dir: Path,
        domains: Mapping[str, CoverageDomain],
    ) -> Self:
        """Rebuild the index by parsing every ``records/<triplet>.geojson``.

        Per-dataset coverage is computed here against ``domains`` (dataset name ->
        :class:`~snowtool.snowdb.coverage.CoverageDomain`) so it stays derived:
        rebuilding the index re-derives it, and a grid/domain change is picked up
        by a plain ``aoi reindex``. Point-only pourpoints are skipped: with no
        basin they are not AOIs (the same rule import applies), and they have no
        geometry hash to index.
        """
        if not records_dir.is_dir():
            return cls({})
        entries = []
        for path in sorted(records_dir.glob('*.geojson')):
            aoi = AOI.from_geojson(path)
            if aoi.polygon is None:
                continue
            entries.append(AOIIndexEntry.from_aoi(aoi, domains))
        return cls.from_entries(entries)

    @classmethod
    def load(cls: type[Self], path: Path) -> Self:
        """Load a persisted ``index.geojson`` (empty index if the file is absent)."""
        if not path.is_file():
            return cls({})
        collection = json.loads(path.read_text())
        return cls.from_entries(
            AOIIndexEntry.from_feature(feature)
            for feature in collection['features']
        )

    def to_feature_collection(self: Self) -> dict[str, Any]:
        return {
            'type': 'FeatureCollection',
            'features': [
                self.entries[triplet].to_feature()
                for triplet in sorted(self.entries)
            ],
        }

    def save(self: Self, path: Path) -> None:
        """Write the index as a sorted, indented ``FeatureCollection``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_feature_collection(), indent=2)
        path.write_text(f'{text}\n')

    def triplets(self: Self) -> set[types.StationTriplet]:
        return set(self.entries)

    def __getitem__(self: Self, triplet: types.StationTriplet) -> AOIIndexEntry:
        return self.entries[triplet]

    def __contains__(self: Self, triplet: types.StationTriplet) -> bool:
        return triplet in self.entries

    def __iter__(self: Self) -> Iterator[AOIIndexEntry]:
        for triplet in sorted(self.entries):
            yield self.entries[triplet]

    def __len__(self: Self) -> int:
        return len(self.entries)
