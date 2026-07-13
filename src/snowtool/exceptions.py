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

    Closes the silent-partial-stats gap: a basin that spills outside (or sits
    entirely off) a dataset's grid would otherwise return zonal stats over only
    its in-grid portion with no warning. Carries the triplet, dataset name, and
    computed ``coverage`` so the caller can report which case fired. ``partial``
    coverage is overridable (``allow_partial`` -- a knowingly-clipped query);
    ``none`` is never allowed, as an off-grid basin has no pixels to compute.
    """

    def __init__(self, triplet: str, dataset: str, coverage: object) -> None:
        self.triplet = triplet
        self.dataset = dataset
        self.coverage = coverage
        # ``coverage`` is a Coverage enum; matched by value to avoid importing it
        # here (coverage.py imports this module for the guard).
        if getattr(coverage, 'value', coverage) == 'partial':
            detail = (
                'is only partially covered by it (the basin extends outside the '
                'grid); pass allow_partial to query the in-grid portion only'
            )
        else:
            detail = 'is not covered by it (the basin is entirely outside the grid)'
        super().__init__(f'Pourpoint {triplet!r} {detail} (dataset {dataset!r}).')


class GeometryOutsideGridError(SnowtoolError, ValueError):
    """Raised when a geometry's bounding box does not intersect a dataset grid.

    Rasterizing a basin burns its tile window onto the dataset grid; a basin
    lying entirely outside the grid has no window (its coverage is ``NONE``), so
    a direct ``Dataset.rasterize_aoi`` surfaces this typed condition instead of
    a degenerate (inverted) window and a numpy shape error. The batch paths
    (``stage_dataset``/``rasterize_aois``) pre-filter by coverage and *skip*
    such basins rather than raising.
    """


class PourpointNotFoundError(SnowtoolError, FileNotFoundError):
    """Raised when no stored pourpoint record exists for a requested triplet.

    A *client* error (the caller referenced a pourpoint not in the database),
    distinct from a bare ``FileNotFoundError`` (a missing file the server expected
    -- a 500). Subclasses ``FileNotFoundError`` so existing ``except
    FileNotFoundError`` call sites (and the CLI) keep catching it, while the HTTP
    API maps *only* this type to 404 and lets a generic ``FileNotFoundError`` 500.
    """


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
    """Raised when a dataset name cannot be used as a bare token/directory name.

    Registration is the single choke point for names, so a name containing a
    path separator or ending in ``.json`` (which ``resolve_dataset``'s syntactic
    partition would read as a path) is rejected there with this type.
    """


class UnknownZoneLayerProviderError(SnowtoolError, ValueError):
    """Raised when a zone-layer provider name matches no configured provider.

    Covers both a ``--provider`` selection and a ``--source PROVIDER PATH``
    override naming a provider the database does not configure.
    """


class ArtifactExistsError(SnowtoolError, FileExistsError):
    """Raised when a write would clobber an existing derived artifact.

    The shared refuse-to-overwrite guard (dataset skeletons, AOI rasters, zone
    layers, ingested COGs): the artifact is already there and the caller did not
    pass ``force``. Subclasses ``FileExistsError`` so existing ``except
    FileExistsError`` call sites (e.g. the tolerate-already-staged path in
    ``stage_dataset``) keep catching it.
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

    ``dataset``/``date``/``missing_keys`` locate the affected data; all three are
    optional so the same type can also cover a corrupt burned-AOI raster (missing
    tile metadata), which has no variable set. Prefer :meth:`for_variables` for the
    common date-is-missing/duplicated-variables case.
    """

    def __init__(
        self: Self,
        detail: str,
        *,
        dataset: str | None = None,
        date: object | None = None,
        missing_keys: Iterable[str] | None = None,
    ) -> None:
        self.dataset = dataset
        self.date = date
        self.missing_keys = sorted(missing_keys) if missing_keys is not None else None
        super().__init__(detail)

    @classmethod
    def for_variables(
        cls: type[Self],
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
            dataset=dataset,
            date=date,
            missing_keys=keys,
        )


class RemoteSourceError(SnowtoolError):
    """Raised when a remote pourpoint source cannot be fetched or enumerated.

    Covers the ``http(s)`` import/sync path (``snowtool.snowdb.pourpoint_remote``):
    an HTTP failure fetching a file, a GitHub tree listing that came back truncated
    or empty, or filename collisions that make the flat temp directory ambiguous.
    Distinct from :class:`GeoJSONValidationError` (a *parsed* file that is invalid)
    so the CLI can report a transport/enumeration problem separately from bad data.
    """


class PourpointPruneDestinationRequiredError(SnowtoolError):
    """Raised when ``pourpoint sync`` would remove stored records but has no archive.

    Carries the triplets that would be pruned so the caller can report the count.
    Removal is destructive, so it is gated behind an explicit ``--prune-to``
    archive destination (or ``--dry-run`` to preview without removing).
    """

    def __init__(self, triplets: list[str]) -> None:
        self.triplets = list(triplets)
        super().__init__(
            f'{len(self.triplets)} stored pourpoint(s) would be removed; pass '
            '--prune-to ARCHIVE to archive them first, or --dry-run to preview.',
        )
