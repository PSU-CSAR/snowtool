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

The on-disk ``Feature``/``FeatureCollection`` shape is geojson-pydantic's, with
one local ``_IndexFeatureProperties`` model for the ``properties`` block -- the
only genuinely index-specific schema here. geojson-pydantic's ``Feature`` allows
a null ``id``/``geometry``/``properties`` (valid GeoJSON, but not a valid index
entry), so the load path guards those into clear errors.

Maintenance is split two ways: import/sync/remove update the index *incrementally*
(``SnowDbManager._update_index`` reuses an entry as-is while its record and the
registered-dataset set are unchanged), while ``pourpoint reindex`` is the explicit
FULL rebuild that ignores the persisted index -- the recovery path for
out-of-band ``records/`` edits and for a grid change to an already-registered
dataset name. Both paths share one loop, :meth:`PourpointIndex.build`: a full
rebuild calls it with no ``reuse``/``preparsed`` (so every record is parsed from
disk), while the incremental path passes the previous index as ``reuse`` and any
just-parsed pourpoints as ``preparsed``.
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from geojson_pydantic import Feature, FeatureCollection, Point
from pydantic import BaseModel, ConfigDict, Field

from snowtool import types
from snowtool.exceptions import IndexedPourpointMissingBasinError
from snowtool.snowdb import triplet_naming
from snowtool.snowdb.atomic import atomic_write_text
from snowtool.snowdb.coverage import Coverage, dataset_coverage
from snowtool.snowdb.pourpoint import Pourpoint
from snowtool.snowdb.progress import NULL_PROGRESS

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from snowtool.snowdb.coverage import CoverageDomain
    from snowtool.snowdb.progress import ProgressReporter


class _IndexFeatureProperties(BaseModel):
    """The index manifest ``Feature``'s ``properties`` block."""

    name: str
    area_meters: float
    geometry_hash: str
    coverage: dict[str, Coverage] = Field(default_factory=dict)


_IndexFeature = Feature[Point, _IndexFeatureProperties]
_IndexFeatureCollection = FeatureCollection[_IndexFeature]


class PourpointIndexEntry(BaseModel):
    """One pourpoint's denormalized list-fields (a single manifest ``Feature``)."""

    model_config = ConfigDict(frozen=True)

    triplet: types.StationTriplet
    name: str
    point: Point
    # Internal rebuild signal (the basin polygon's WKB hash); not surfaced by the
    # API, kept so a stale index can be detected/rebuilt.
    geometry_hash: str
    # Geodesic basin area (m^2), computed from the polygon at reindex so the list
    # can report it without parsing the (large) basin records. Always present: an
    # index entry only exists for a basin-bearing pourpoint (point-only ones are
    # never indexed), so the area is always computable.
    area_meters: float
    # Per-dataset geometric coverage of this pourpoint's basin, keyed by dataset
    # name. Derived: a full rebuild (`PourpointIndex.build` with no `reuse`)
    # recomputes it, registration folds a new dataset's key in, and incremental
    # import/sync/remove reuse an entry only while its key set still matches the
    # registered datasets -- so a grid change to an already-registered name
    # needs a `pourpoint reindex`.
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
            type='Feature',
            id=self.triplet,
            geometry=self.point,
            properties=_IndexFeatureProperties(
                name=self.name,
                area_meters=self.area_meters,
                geometry_hash=self.geometry_hash,
                coverage=self.coverage,
            ),
        )

    @classmethod
    def _from_index_feature(cls: type[Self], feature: _IndexFeature) -> Self:
        # geojson-pydantic permits a null id/geometry/properties (valid GeoJSON);
        # an index entry needs all three, so a foreign/corrupt feature fails
        # loudly here -- the index is derived, so the fix is `pourpoint reindex`.
        if feature.id is None or feature.geometry is None or feature.properties is None:
            raise ValueError(
                'index feature is missing its id, geometry, or properties; '
                'rebuild the index with `pourpoint reindex`.',
            )
        return cls(
            triplet=str(feature.id),
            name=feature.properties.name,
            point=feature.geometry,
            geometry_hash=feature.properties.geometry_hash,
            area_meters=feature.properties.area_meters,
            coverage=feature.properties.coverage,
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
    def build(
        cls: type[Self],
        paths: Iterable[Path],
        domains: Mapping[str, CoverageDomain],
        *,
        reuse: Mapping[types.StationTriplet, PourpointIndexEntry] = {},
        preparsed: Mapping[types.StationTriplet, Pourpoint] = {},
        progress: ProgressReporter = NULL_PROGRESS,
    ) -> Self:
        """Build an index over ``paths`` (one ``records/<triplet>.geojson`` each).

        The one loop shared by the explicit FULL rebuild (``pourpoint reindex``,
        called with no ``reuse``/``preparsed`` -- every record is parsed from
        disk) and the incremental update after an import/sync/remove (called
        with the previous index as ``reuse`` and this operation's just-parsed
        pourpoints as ``preparsed``). Per record, in order: a triplet present in
        ``preparsed`` is indexed from that in-memory :class:`Pourpoint` (no disk
        re-parse); else a triplet present in ``reuse`` whose entry's coverage keys
        still equal ``domains`` is kept as-is; else the record is parsed from disk
        and indexed. Every stored record is basin-bearing (the import boundary
        guarantees it), so a basin-less parsed record is a corrupt store and
        raises :class:`IndexedPourpointMissingBasinError` naming the offending
        file rather than being silently dropped. Per-dataset coverage for a
        freshly indexed
        entry is computed against ``domains`` (dataset name ->
        :class:`~snowtool.snowdb.coverage.CoverageDomain`) so it stays derived:
        a full rebuild re-derives it, and a grid/domain change is picked up by a
        plain ``pourpoint reindex`` -- the one change a ``reuse`` hit cannot see
        (its coverage keys match but the domain behind an unchanged name may have
        moved). ``progress`` reports the pass, advancing once per path whether
        reused, indexed from memory, or parsed.
        """
        paths = sorted(paths)
        entries: list[PourpointIndexEntry] = []
        with progress.track(
            f'indexing {len(paths)} pourpoint(s)',
            total=len(paths),
        ) as task:
            for path in paths:
                triplet = triplet_naming.stem_to_triplet(path.stem)
                if triplet in preparsed:
                    entries.append(
                        PourpointIndexEntry.from_pourpoint(preparsed[triplet], domains),
                    )
                elif triplet in reuse and set(reuse[triplet].coverage) == set(domains):
                    entries.append(reuse[triplet])
                else:
                    pourpoint = Pourpoint.from_geojson(path)
                    if pourpoint.polygon is None:
                        raise IndexedPourpointMissingBasinError(
                            f'Corrupt pourpoint store: record {path.name} has no '
                            f'basin polygon (every stored record must be '
                            f'basin-bearing).',
                        )
                    entries.append(
                        PourpointIndexEntry.from_pourpoint(pourpoint, domains),
                    )
                task.advance()
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
            type='FeatureCollection',
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
