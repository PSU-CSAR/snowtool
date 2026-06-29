from __future__ import annotations

from typing import Self

from gazebo.link import Link
from pydantic import BaseModel, Field

from snowtool import __version__


class VersionInfo(BaseModel):
    version: str = __version__
    links: list[Link] = Field(default_factory=list)

    @classmethod
    def build(cls) -> Self:
        return cls(links=[Link.self_link(), Link.root_link()])
