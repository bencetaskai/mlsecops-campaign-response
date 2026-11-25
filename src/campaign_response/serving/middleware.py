"""Per-request context: a request id on every log line and in the response, and one timing line per request."""

import logging
import re
import time
import uuid
from typing import Final

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from campaign_response.logging_setup import correlation_id_var

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER: Final = "X-Request-ID"
# A caller's id is written to the logs, so only short plain tokens are reused; anything else (a newline could
# forge a log line) is replaced by a fresh id.
_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


def request_id_from(header: str | None) -> str:
    """The caller's id if it is safe to log, otherwise a new random one."""
    if header is not None and _SAFE_REQUEST_ID.fullmatch(header):
        return header
    return uuid.uuid4().hex[:16]


async def request_context(request: Request, call_next: RequestResponseEndpoint) -> Response:
    """Set the request id for the whole request, echo it in the response and log method, path, status and time.

    The path is logged without the query string, which may carry data. An unhandled error becomes a generic 500
    with the request id, so the caller can quote the id while the details stay in the log.
    """
    request_id = request_id_from(request.headers.get(REQUEST_ID_HEADER))
    request.state.request_id = request_id
    token = correlation_id_var.set(request_id)
    started = time.perf_counter()
    try:
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("unhandled error")
            response = JSONResponse(
                {"detail": "internal server error", "request_id": request_id},
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info("request finished", extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        })
        return response
    finally:
        correlation_id_var.reset(token)
