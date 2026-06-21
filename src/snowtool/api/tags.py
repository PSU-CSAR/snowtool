from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class TagDocs(BaseModel, frozen=True):
    description: str
    url: HttpUrl


class Tag(BaseModel, frozen=True):
    name: str
    description: str | None = None
    external_docs: TagDocs | None = Field(
        default=None,
        serialization_alias='externalDocs',
    )


class Tags(StrEnum):
    ROOT = 'root'
    DATASETS = 'datasets'

    def __init__(
        self,
        name: str,
        description: str | None = None,
        external_docs: TagDocs | None = None,
    ) -> None:
        self.tag = Tag(
            name=name,
            description=description,
            external_docs=external_docs,
        )

    @classmethod
    def metadata(cls) -> list[dict[str, Any]]:
        return [member.tag.model_dump(exclude_none=True) for member in cls]
