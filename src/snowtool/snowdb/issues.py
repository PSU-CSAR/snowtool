"""Typed validation issues shared by the write path (skip-unless-issues) and the
read path (`doctor` reporting).

An ``Issue`` is one thing wrong with (or worth noting about) an artifact. It is
``isinstance``-checkable (no prose matching in code), renders itself to a human
string via ``message``, and declares whether re-writing the artifact resolves it
(``actionable``). A check returns ``list[Issue]`` -- empty means current/healthy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class Issue:
    """One validation problem. Subclasses set fields and override ``message``."""

    @property
    def actionable(self) -> bool:
        """Whether re-writing/rebuilding the artifact resolves this issue."""
        return False

    @property
    def message(self) -> str:  # pragma: no cover - overridden by every subclass
        raise NotImplementedError

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class GridMismatch(Issue):
    declared: tuple[float, ...] = ()
    actual: tuple[float, ...] = ()

    @property
    def actionable(self) -> bool:
        return True

    @property
    def message(self) -> str:
        return (
            f'declared grid transform {self.declared} does not match '
            f'artifact transform {self.actual}'
        )


@dataclass(frozen=True)
class ShapeMismatch(Issue):
    declared: tuple[int, int] = (0, 0)
    actual: tuple[int, int] = (0, 0)

    @property
    def actionable(self) -> bool:
        return True

    @property
    def message(self) -> str:
        dcols, drows = self.declared[1], self.declared[0]
        return (
            f'declared grid is {dcols}x{drows} (cols x rows) but artifact is '
            f'{self.actual[1]}x{self.actual[0]}'
        )


@dataclass(frozen=True)
class ContentStale(Issue):
    stored: str | None = None
    expected: str = ''

    @property
    def actionable(self) -> bool:
        return True

    @property
    def message(self) -> str:
        return f'stale content (stored {self.stored} != expected {self.expected})'


@dataclass(frozen=True)
class FormatStale(Issue):
    stored: int | None = None
    current: int = 0

    @property
    def actionable(self) -> bool:
        return True

    @property
    def message(self) -> str:
        return f'stale format (stored {self.stored} != current {self.current})'


@dataclass(frozen=True)
class Unreadable(Issue):
    detail: str = ''

    @property
    def message(self) -> str:
        return f'unreadable: {self.detail}'


@dataclass(frozen=True)
class EmptyArtifact(Issue):
    @property
    def message(self) -> str:
        return 'empty AOI raster (covers no in-grid cells: off-grid or masked)'


@dataclass(frozen=True)
class MissingProvenanceTag(Issue):
    tag: str = ''

    @property
    def actionable(self) -> bool:
        return True

    @property
    def message(self) -> str:
        return f'missing {self.tag} tag (rebuild with `pourpoint rasterize --rebuild`)'


@dataclass(frozen=True)
class OrphanArtifact(Issue):
    @property
    def message(self) -> str:
        return 'orphan raster'


@dataclass(frozen=True)
class MissingArtifact(Issue):
    @property
    def actionable(self) -> bool:
        return True

    @property
    def message(self) -> str:
        return 'missing'


@dataclass(frozen=True)
class NoCoverage(Issue):
    @property
    def message(self) -> str:
        return 'no coverage'


@dataclass(frozen=True)
class PartialCoverage(Issue):
    @property
    def message(self) -> str:
        return 'partial coverage'


@dataclass(frozen=True)
class NoRaster(Issue):
    @property
    def actionable(self) -> bool:
        return True

    @property
    def message(self) -> str:
        return 'no raster'


@dataclass(frozen=True)
class UnverifiableFreshness(Issue):
    reason: str = ''

    @property
    def message(self) -> str:
        return f'freshness unverifiable: {self.reason}'


def render(issues: Iterable[Issue]) -> str:
    """Join issue messages with '; ' (doctor's one-row-per-target rendering)."""
    return '; '.join(issue.message for issue in issues)
