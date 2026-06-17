from __future__ import annotations

from fastapi import Request, status
from fastapi.responses import JSONResponse


class APIError(Exception):
    http_status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def to_json(self) -> JSONResponse:
        content = {
            'error_type': self.__class__.__name__,
            'status': self.http_status_code,
            'detail': str(self),
        }
        return JSONResponse(
            status_code=self.http_status_code,
            content=content,
        )


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    if exc.http_status_code >= 500:
        request.app.state.logger.exception('Server Error')
    return exc.to_json()


class NotFoundError(APIError):
    http_status_code: int = status.HTTP_404_NOT_FOUND


class ValidationError(APIError):
    http_status_code: int = status.HTTP_422_UNPROCESSABLE_ENTITY
