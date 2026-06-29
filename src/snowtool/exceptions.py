class SNODASError(Exception):
    pass


class SNODASWarning(UserWarning):
    """Base class for snowtool warnings (suspect-but-not-fatal conditions).

    A distinct category so callers can filter or escalate snowtool's warnings
    specifically (e.g. ``warnings.simplefilter('error', SNODASWarning)``).

    Convention: ``warnings.warn(..., SNODASWarning)`` for suspect data/state
    conditions the operation tolerates (escalatable, deduped per call site);
    ``logging`` for operational progress and errors.
    """


class GeoJSONValidationError(SNODASError, TypeError):
    pass


class SnowDbConfigError(SNODASError):
    """Raised when a snowdb root has no (or an invalid) root config file.

    The root config is the system's single entry point, so a root lacking one is
    not a snowdb this version understands -- there is no lenient un-initialized
    read path (the deliberate no-backwards-compat call). Carries the root that was
    opened and points the operator at ``snowtool migration stamp`` to write a
    config for a legacy layout.
    """

    def __init__(self, root: object, detail: str | None = None) -> None:
        self.root = root
        message = detail or (
            f'{root} is not a snowdb (no root config). Run '
            '`snowtool migration stamp` to write one for a legacy layout, or '
            '`snowtool snowdb init` to create a new root.'
        )
        super().__init__(message)


class AOICoverageError(SNODASError):
    """Raised when an AOI is queried against a dataset that does not fully cover it.

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
        super().__init__(f'AOI {triplet!r} {detail} (dataset {dataset!r}).')


class AOINotFoundError(SNODASError, FileNotFoundError):
    """Raised when no stored AOI record exists for a requested triplet.

    A *client* error (the caller referenced an AOI that is not in the database),
    distinct from a bare ``FileNotFoundError`` (a missing file the server expected
    -- a 500). Subclasses ``FileNotFoundError`` so existing ``except
    FileNotFoundError`` call sites (and the CLI) keep catching it, while the HTTP
    API maps *only* this type to 404 and lets a generic ``FileNotFoundError`` 500.
    """


class AOIRasterNotFoundError(SNODASError, FileNotFoundError):
    """Raised when an AOI's burned raster has not been built for a dataset.

    A missing prerequisite the caller can fix (``aoi rasterize``), so the HTTP API
    maps it to 404 rather than letting it 500. See :class:`AOINotFoundError` for the
    ``FileNotFoundError`` subclassing rationale.
    """


class QueryParameterError(SNODASError, ValueError):
    """Raised for an invalid query parameter (unknown variable/zone, runaway cross).

    A *client* error in the stats/zonal query surface -- an unknown variable or zone
    layer, an unparseable ``--zone`` override, or a crossed query exceeding
    ``max_zone_cells``. Subclasses ``ValueError`` so existing ``except ValueError``
    call sites (and the CLI) keep catching it, while the HTTP API maps *only* this
    type to 422 and lets a generic ``ValueError`` (a real bug) 500.
    """


class AOIPruneDestinationRequiredError(SNODASError):
    """Raised when ``aoi sync`` would remove stored AOIs but has no archive dir.

    Carries the triplets that would be pruned so the caller can report the count.
    Removal is destructive, so it is gated behind an explicit ``--prune-to``
    archive destination (or ``--dry-run`` to preview without removing).
    """

    def __init__(self, triplets: list[str]) -> None:
        self.triplets = list(triplets)
        super().__init__(
            f'{len(self.triplets)} stored AOI(s) would be removed; pass '
            '--prune-to ARCHIVE to archive them first, or --dry-run to preview.',
        )


class LedgerError(SNODASError):
    """
    Raised when accessing the ledger for tracking failed download attempts fails
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class DownloadError(SNODASError):
    """
    Raised when a Download request for a data file results
    in an internal server error
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
