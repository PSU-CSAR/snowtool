"""The derived pourpoint manifest: a GeoJSON ``FeatureCollection`` index of records.

The per-pourpoint geojson under ``pourpoints/records/`` are the lossless source of
truth, but parsing every one (each basin is thousands of coordinate pairs) just to
list the pourpoints is wasteful. :class:`PourpointIndex` is a lightweight,
rebuildable index of the list-relevant fields -- one ``Point`` ``Feature`` per
pourpoint (``id`` = triplet, ``geometry`` = the pourpoint point, ``properties`` =
name + geodesic ``area_meters`` + per-dataset coverage, plus the basin
:attr:`~snowtool.snowdb.pourpoint.Pourpoint.geometry_hash` as an internal rebuild
signal) -- persisted to ``pourpoints/index.geojson``. It is GeoJSON-native on
purpose: the same file is a plottable point layer and the FastAPI listing payload.
Being derived, it is always rebuildable from ``records/`` (``pourpoint reindex``),
so it never has to be trusted as primary data.
"""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from snowtool import types
from snowtool.snowdb.atomic import atomic_write_text
from snowtool.snowdb.coverage import Coverage, dataset_coverage
from snowtool.snowdb.pourpoint import Pourpoint

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from snowtool.snowdb.coverage import CoverageDomain


@dataclass(frozen=True)
class PourpointIndexEntry:
    """One pourpoint's denormalized list-fields (a single manifest ``Feature``)."""

    triplet: types.StationTriplet
    name: str
    point: dict[str, Any]
    # Internal rebuild signal (the basin polygon's WKB hash); not surfaced by the
    # API, kept so a stale index can be detected/rebuilt.
    geometry_hash: str
    # Geodesic basin area (m^2), computed from the polygon at reindex so the list
    # can report it without parsing the (large) basin records.
    area_meters: float | None = None
    # Per-dataset geometric coverage of this pourpoint's basin, keyed by dataset
    # name. Derived (a grid change => `pourpoint reindex`), so it is recomputed by
    # `PourpointIndex.from_records`, never edited in place.
    coverage: dict[str, Coverage] = field(default_factory=dict)

    @classmethod
    def from_pourpoint(
        cls: type[Self],
        pourpoint: Pourpoint,
        domains: Mapping[str, CoverageDomain],
    ) -> Self:
        return cls(
            triplet=pourpoint.station_triplet,
            name=pourpoint.name,
            point=pourpoint.point,
            geometry_hash=pourpoint.geometry_hash,
            area_meters=pourpoint.area_meters,
            coverage={
                name: dataset_coverage(pourpoint, domain)
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
                'area_meters': self.area_meters,
                'geometry_hash': self.geometry_hash,
                'coverage': {name: cov.value for name, cov in self.coverage.items()},
            },
        }

    @classmethod
    def from_feature(cls: type[Self], feature: dict[str, Any]) -> Self:
        properties = feature['properties']
        return cls(
            triplet=types.StationTriplet(feature['id']),
            name=properties['name'],
            point=feature['geometry'],
            geometry_hash=properties['geometry_hash'],
            area_meters=properties.get('area_meters'),
            coverage={
                name: Coverage(value) for name, value in properties['coverage'].items()
            },
        )


@dataclass
class PourpointIndex:
    """The set of :class:`PourpointIndexEntry`, keyed by triplet.

    Serializes to / from a GeoJSON ``FeatureCollection`` (features sorted by
    triplet for stable, reviewable diffs). Build it from the ``records/`` dir
    (the rebuild path) or load it from a persisted ``index.geojson``.
    """

    entries: dict[types.StationTriplet, PourpointIndexEntry]

    @classmethod
    def from_entries(cls: type[Self], entries: Iterable[PourpointIndexEntry]) -> Self:
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
        by a plain ``pourpoint reindex``. Point-only pourpoints are skipped: with no
        basin they have nothing to cover and no geometry hash to index.
        """
        if not records_dir.is_dir():
            return cls({})
        entries = []
        for path in sorted(records_dir.glob('*.geojson')):
            pourpoint = Pourpoint.from_geojson(path)
            if pourpoint.polygon is None:
                continue
            entries.append(PourpointIndexEntry.from_pourpoint(pourpoint, domains))
        return cls.from_entries(entries)

    @classmethod
    def load(cls: type[Self], path: Path) -> Self:
        """Load a persisted ``index.geojson`` (empty index if the file is absent)."""
        if not path.is_file():
            return cls({})
        collection = json.loads(path.read_text())
        return cls.from_entries(
            PourpointIndexEntry.from_feature(feature)
            for feature in collection['features']
        )

    def to_feature_collection(self: Self) -> dict[str, Any]:
        return {
            'type': 'FeatureCollection',
            'features': [
                self.entries[triplet].to_feature() for triplet in sorted(self.entries)
            ],
        }

    def save(self: Self, path: Path) -> None:
        """Write the index as a sorted, indented ``FeatureCollection``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_feature_collection(), indent=2)
        atomic_write_text(path, f'{text}\n')

    def triplets(self: Self) -> set[types.StationTriplet]:
        return set(self.entries)

    def __getitem__(self: Self, triplet: types.StationTriplet) -> PourpointIndexEntry:
        return self.entries[triplet]

    def __contains__(self: Self, triplet: types.StationTriplet) -> bool:
        return triplet in self.entries

    def __iter__(self: Self) -> Iterator[PourpointIndexEntry]:
        for triplet in sorted(self.entries):
            yield self.entries[triplet]

    def __len__(self: Self) -> int:
        return len(self.entries)
