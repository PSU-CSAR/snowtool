from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DownloadResult:
    url: str
    dest: Path
    status: str  # "downloaded", "skipped_exists", "missing", "error", "verify_failed"
    detail: str = ''
