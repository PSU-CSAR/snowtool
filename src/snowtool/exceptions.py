class SNODASError(Exception):
    pass


class GeoJSONValidationError(SNODASError, TypeError):
    pass


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
