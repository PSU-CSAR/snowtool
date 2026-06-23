class SNODASError(Exception):
    pass


class SNODASWarning(UserWarning):
    """Base class for snowtool warnings (suspect-but-not-fatal conditions).

    A distinct category so callers can filter or escalate snowtool's warnings
    specifically (e.g. ``warnings.simplefilter('error', SNODASWarning)``).
    """


class GeoJSONValidationError(SNODASError, TypeError):
    pass


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
