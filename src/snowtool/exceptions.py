from __future__ import annotations

from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from collections.abc import Iterable


class SnowtoolError(Exception):
    pass


class SnowtoolWarning(UserWarning):
    """Base class for snowtool warnings (suspect-but-not-fatal conditions).

    A distinct category so callers can filter or escalate snowtool's warnings
    specifically (e.g. ``warnings.simplefilter('error', SnowtoolWarning)``).

    Convention: ``warnings.warn(..., SnowtoolWarning)`` for suspect data/state
    conditions the operation tolerates (escalatable, deduped per call site);
    ``logging`` for operational progress and errors.
    """


class GeoJSONValidationError(SnowtoolError, TypeError):
    pass


class SnowDbConfigError(SnowtoolError):
    """Raised when a snowdb root has no (or an invalid) root config file.

    The root config is the system's single entry point, so a root lacking one is
    not a snowdb this version understands -- there is no lenient un-initialized
    read path (the deliberate no-backwards-compat call). Carries the root that was
    opened and points the operator at ``snowtool init`` to create one.
    """

    def __init__(self, root: object, detail: str | None = None) -> None:
        self.root = root
        message = detail or (
            f'{root} is not a snowdb (no root config). '
            'Run `snowtool init` to create one.'
        )
        super().__init__(message)


class PourpointCoverageError(SnowtoolError):
    """Raised when a pourpoint is queried against a dataset not fully covering it.

    A basin that spills outside (or sits entirely off) a dataset's grid would
    otherwise return zonal stats over only its in-grid portion with no warning.
    Carries the triplet, dataset name, and computed ``coverage`` so the caller
    can report which case fired. ``partial`` coverage is overridable
    (``allow_partial`` -- a knowingly-clipped query); ``none`` is never
    allowed, as an off-grid basin has no pixels to compute.

    A plain carrier: :func:`~snowtool.snowdb.coverage.require_full_coverage` owns
    the ``Coverage`` enum and builds the rendered ``message``, so this class does
    not need to know its members (or duck-type around them).
    """

    def __init__(
        self,
        triplet: str,
        dataset: str,
        coverage: object,
        message: str,
    ) -> None:
        self.triplet = triplet
        self.dataset = dataset
        self.coverage = coverage
        super().__init__(message)


class GeometryOutsideGridError(SnowtoolError, ValueError):
    """Raised when a geometry's bounding box does not intersect a dataset grid.

    Rasterizing a basin burns its tile window onto the dataset grid; a basin
    lying entirely outside the grid has no window (its coverage is ``NONE``), so
    a direct ``Dataset.rasterize_aoi`` surfaces this typed condition instead of
    a degenerate (inverted) window and a numpy shape error. The batch path
    (``rasterize_aois``, which ``stage_dataset`` also goes through) checks each
    pourpoint-dataset pair's coverage and *skips* such basins rather than
    raising.
    """


class PourpointNotFoundError(SnowtoolError, FileNotFoundError):
    """Raised when no stored pourpoint record exists for a requested triplet.

    A *client* error (the caller referenced a pourpoint not in the database),
    distinct from a bare ``FileNotFoundError`` (a missing file the server expected
    -- a 500). Subclasses ``FileNotFoundError`` so existing ``except
    FileNotFoundError`` call sites (and the CLI) keep catching it, while the HTTP
    API maps *only* this type to 404 and lets a generic ``FileNotFoundError`` 500.
    """

    @classmethod
    def for_triplet(cls, triplet: object) -> Self:
        """The canonical 'no such stored pourpoint' error for a ``triplet``."""
        return cls(f'No stored pourpoint for triplet {triplet!r}.')


class AOIRasterNotFoundError(SnowtoolError, FileNotFoundError):
    """Raised when an AOI's burned raster has not been built for a dataset.

    A missing prerequisite the caller can fix (``pourpoint rasterize``), so the API
    maps it to 404 rather than letting it 500. See
    :class:`PourpointNotFoundError` for the ``FileNotFoundError`` subclassing
    rationale.
    """


class QueryParameterError(SnowtoolError, ValueError):
    """Raised for an invalid parameter on a read surface (stats, diagnostics).

    A *client* error in a parameterized query -- an unknown variable or zone
    layer, an unparseable ``--zone`` override or ``--dates`` interval, a crossed
    query exceeding ``max_zone_cells``, or a date-range request with no
    inferable start. Subclasses ``ValueError`` so existing ``except ValueError``
    call sites (and the CLI) keep catching it, while the HTTP API maps *only* this
    type to 422 and lets a generic ``ValueError`` (a real bug) 500.
    """


class ZoneParamsError(SnowtoolError, ValueError):
    """Raised when a dataset's ``zones`` block configures a layer with params of
    the wrong kind (e.g. ``buckets`` for the banded elevation axis).

    Zone params parse to a *specific* member model at config load (an unknown
    param name already fails there); this guards the remaining gap -- a
    well-formed param attached to a layer whose scheme doesn't take it -- which
    is only detectable once the layer's scheme is known. A dataset-config
    (operator) error, deliberately not a :class:`QueryParameterError`: the API
    must not report it as a client 422.
    """


class NodataMaskError(SnowtoolError):
    """Raised when a dataset's configured nodata mask cannot be used.

    A dataset-config (operator) error, like :class:`ZoneParamsError`: the
    config's ``nodata_mask`` names a file that is missing, or the file's
    raster shape does not match the dataset grid (the mask window is read by
    pixel offsets, so a mismatched raster would silently misalign -- refuse it
    instead). The fix is restoring/correcting the mask file or removing
    ``nodata_mask`` from the dataset config, never issuing a different
    request, so the API must not map it to a client error.
    """


class UnknownDatasetError(SnowtoolError, ValueError):
    """Raised when a dataset token resolves to nothing actionable.

    A management-surface *client* error: a name the root config does not
    register (``resolve_dataset``, ``set_dataset_active``) or a path token with
    no dataset config file behind it. Subclasses ``ValueError`` so existing
    ``except ValueError`` call sites keep catching it.
    """


class InvalidDatasetNameError(SnowtoolError, ValueError):
    """Raised when a dataset name reads as a path token (see
    ``snowtool.snowdb.manager._is_path_token``); rejected at registration, the
    single choke point for names.
    """


class UnknownZoneLayerProviderError(SnowtoolError, ValueError):
    """Raised when a zone-layer provider name matches no configured provider.

    Covers both a ``--provider`` selection and a ``--source PROVIDER PATH``
    override naming a provider the database does not configure.
    """


class UnknownHealthCheckError(SnowtoolError, ValueError):
    """Raised when ``doctor`` is asked to run a check name it does not know.

    ``run_health_checks`` validates its requested names against the registry, so
    an unknown name surfaces here (rendered as a clean CLI usage line) rather
    than as a bare ``KeyError`` traceback.
    """


class ArtifactExistsError(SnowtoolError):
    """Raised when a zone-layer generation would clobber an existing layer.

    The zone-layer generators (terrain, land cover) refuse to overwrite a layer
    that is already present unless the caller passes ``--force``; this is that
    refusal.
    """


class IngestSourceError(SnowtoolError, ValueError):
    """Raised when an ingest source artifact does not have the expected shape.

    An operator-facing *input* error -- a file that is not the dataset's source
    format (an unparseable SNODAS member name, a non-header raster path) --
    distinct from :class:`ArtifactExistsError` (output already present) and from
    a bare ``ValueError`` (a real bug).
    """


class IncompleteDatasetDataError(SnowtoolError):
    """Raised when a dataset's on-disk data for a date is incomplete or unresolvable.

    A server-side *data-integrity* failure, not a client error: a date directory is
    missing one or more of the dataset's required variable COGs, or a variable
    resolves to more than one file -- a partial/crashed ingest, a deleted COG, or a
    stale duplicate left behind by a differently-named source (an INSTARR version
    bump, a SWANN stage change). Distinct from a bare ``ValueError``/``RuntimeError``
    so the read path can surface it as a clean, typed condition while a genuine bug
    still 500s generically. The message names the dataset, date, and affected
    variable keys but never on-disk paths, so it is safe to relay in an RFC 9457
    problem body. It stays **500-class** on the API (the fix is re-ingesting the
    date, not issuing a different request); the CLI renders it as a clean
    ``ClickException`` via its ``SnowtoolError`` base.

    The same type also covers a corrupt burned-AOI raster (missing tile
    metadata), which has no variable set. Prefer :meth:`for_variables` for the
    common date-is-missing/duplicated-variables case.
    """

    @classmethod
    def for_variables(
        cls,
        dataset: str,
        date: object,
        missing_keys: Iterable[str],
    ) -> Self:
        """Build the common 'date cannot resolve some variables' error.

        ``missing_keys`` are the variable keys that matched no COG (missing) or
        more than one (duplicated) -- both leave a date's data incomplete.
        """
        keys = sorted(missing_keys)
        return cls(
            f'Incomplete data for {date} in dataset {dataset!r}: cannot resolve '
            f'variable(s) {keys} (missing or duplicated COGs).',
        )


class IndexedPourpointMissingBasinError(SnowtoolError, ValueError):
    """Raised when an *indexed* pourpoint's stored record has no basin polygon.

    The index only lists basin-bearing pourpoints, so loading an indexed
    triplet's basin and finding ``None`` is a data-integrity bug in the stored
    record (an out-of-band ``records/`` edit not followed by ``pourpoint
    reindex``), not a client error. Deliberately **not** registered in
    ``api/exceptions.py``, so -- like a bare ``ValueError`` -- it surfaces as a
    genuine server 500 rather than a mapped client problem. Subclasses
    ``ValueError`` so existing ``except ValueError`` call sites keep catching it.
    """


class RemoteSourceError(SnowtoolError):
    """Raised when a remote data source cannot be fetched, enumerated, or assembled.

    Covers the ``http(s)`` pourpoint import/sync path (``snowtool.cli._remote``):
    an HTTP failure fetching a file, a GitHub tree listing that came back truncated
    or empty, or filename collisions that make the flat temp directory ambiguous.
    Also the 3DEP DEM source (``snowtool.snowdb.zones.terrain_source``): no tiles
    published for the requested extent, or a discovered tile whose GeoTIFF header
    cannot be turned into a mosaic input (not north-up/georeferenced, an
    unsupported sample format or dtype, or tiles that disagree on
    CRS/resolution/dtype/nodata). All are operator-facing remote-data failures the
    CLI's central mapping renders cleanly. Distinct from
    :class:`GeoJSONValidationError` (a *parsed* file that is invalid) so the CLI can
    report a transport/enumeration problem separately from bad data.
    """


class PourpointPruneDestinationRequiredError(SnowtoolError):
    """Raised when ``pourpoint sync`` would remove stored records but has no archive.

    The message reports how many records would be pruned. Removal is
    destructive, so it is gated behind an explicit ``--prune-to`` archive
    destination (or ``--dry-run`` to preview without removing).
    """

    def __init__(self, triplets: list[str]) -> None:
        super().__init__(
            f'{len(triplets)} stored pourpoint(s) would be removed; pass '
            '--prune-to ARCHIVE to archive them first, or --dry-run to preview.',
        )
