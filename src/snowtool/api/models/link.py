from typing import Annotated, Any, Self

from fastapi import Request
from pydantic import (
    AfterValidator,
    AnyUrl,
    BaseModel,
    Json,
    SerializerFunctionWrapHandler,
    model_serializer,
)


def is_url(value: object) -> AnyUrl:
    return value if isinstance(value, AnyUrl) else AnyUrl(str(value))


class Link(BaseModel, extra='allow'):
    href: Annotated[object, AfterValidator(is_url)]
    rel: str
    type: str | None = None
    title: str | None = None
    method: str | None = None
    headers: dict[str, str | list[str]] | None = None
    body: Json[Any] = None

    # overriding the default serialization to filter None field values from
    # dumped json
    @model_serializer(mode='wrap', when_used='json')
    def serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return {k: v for k, v in handler(self).items() if v is not None}

    @classmethod
    def root_link(cls, request: Request) -> Self:
        return cls(
            href=request.url_for('get_landing_page'),
            rel='root',
            type='application/json',
        )

    @classmethod
    def self_link(cls, request: Request, **kwargs) -> Self:
        attrs = {
            'href': str(request.url),
            'rel': 'self',
            'type': 'application/json',
        }
        attrs.update(**kwargs)
        return cls.model_validate(attrs)
