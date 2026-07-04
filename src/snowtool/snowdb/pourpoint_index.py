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

``_IndexFeature``/``_IndexFeatureCollection`` are minimal local pydantic models for
the on-disk ``Feature``/``FeatureCollection`` shape -- just this module's own
(de)serialization concern, distinct from the API's gazebo-backed response models.
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from snowtool import types
from snowtool.snowdb.atomic import atomic_write_text
from snowtool.snowdb.coverage import Coverage, dataset_coverage
from snowtool.snowdb.geometry import PointGeometry
from snowtool.snowdb.pourpoint import Pourpoint

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from snowtool.snowdb.coverage import CoverageDomain


class _IndexFeatureProperties(BaseModel):
    """The index manifest ``Feature``'s ``properties`` block."""

    name: str
    area_meters: float | None = None
    geometry_hash: str
    coverage: dict[str, Coverage] = Field(default_factory=dict)


class _IndexFeature(BaseModel):
    """One index manifest ``Feature`` -- point geometry + denormalized properties."""

    type: Literal['Feature'] = 'Feature'
    id: types.StationTriplet
    geometry: PointGeometry
    properties: _IndexFeatureProperties


class _IndexFeatureCollection(BaseModel):
    """The persisted ``index.geojson`` -- a plain (link-free) ``FeatureCollection``."""

    type: Literal['FeatureCollection'] = 'FeatureCollection'
    features: list[_IndexFeature]


class PourpointIndexEntry(BaseModel):
    """One pourpoint's denormalized list-fields (a single manifest ``Feature``)."""

    model_config = ConfigDict(frozen=True)

    triplet: types.StationTriplet
    name: str
    point: PointGeometry
    # Internal rebuild signal (the basin polygon's WKB hash); not surfaced by the
    # API, kept so a stale index can be detected/rebuilt.
    geometry_hash: str
    # Geodesic basin area (m^2), computed from the polygon at reindex so the list
    # can report it without parsing the (large) basin records.
    area_meters: float | None = None
    # Per-dataset geometric coverage of this pourpoint's basin, keyed by dataset
    # name. Derived (a grid change => `pourpoint reindex`), so it is recomputed by
    # `PourpointIndex.from_records`, never edited in place.
    coverage: dict[str, Coverage] = Field(default_factory=dict)

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

    def _to_index_feature(self: Self) -> _IndexFeature:
        return _IndexFeature(
            id=self.triplet,
            geometry=self.point,
            properties=_IndexFeatureProperties(
                name=self.name,
                area_meters=self.area_meters,
                geometry_hash=self.geometry_hash,
                coverage=self.coverage,
            ),
        )

    def to_feature(self: Self) -> dict[str, Any]:
        return self._to_index_feature().model_dump(mode='json')

    @classmethod
    def _from_index_feature(cls: type[Self], feature: _IndexFeature) -> Self:
        return cls(
            triplet=feature.id,
            name=feature.properties.name,
            point=feature.geometry,
            geometry_hash=feature.properties.geometry_hash,
            area_meters=feature.properties.area_meters,
            coverage=feature.properties.coverage,
        )

    @classmethod
    def from_feature(cls: type[Self], feature: dict[str, Any]) -> Self:
        return cls._from_index_feature(_IndexFeature.model_validate(feature))


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
        collection = _IndexFeatureCollection.model_validate_json(path.read_text())
        return cls.from_entries(
            PourpointIndexEntry._from_index_feature(feature)
            for feature in collection.features
        )

    def to_feature_collection(self: Self) -> dict[str, Any]:
        collection = _IndexFeatureCollection(
            features=[
                self.entries[triplet]._to_index_feature()
                for triplet in sorted(self.entries)
            ],
        )
        return collection.model_dump(mode='json')

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
