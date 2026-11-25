"""API key check for the endpoints that cost compute or change state (/train, /predict)."""

import hmac
import logging
from collections.abc import Callable
from typing import Annotated, Final

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import SecretStr

logger = logging.getLogger(__name__)

API_KEY_HEADER: Final = "X-API-Key"
INVALID_API_KEY: Final = "missing or invalid API key"

# auto_error=False: a missing header must get the same 401 as a wrong one, not FastAPI's own response.
_api_key_header = APIKeyHeader(
    name=API_KEY_HEADER, auto_error=False, description="Required for /train and /predict.",
)


def api_key_checker(expected: SecretStr) -> Callable[[str | None], None]:
    """A FastAPI dependency that lets a request through only with the expected key."""
    expected_bytes = expected.get_secret_value().encode()

    def check(provided: Annotated[str | None, Security(_api_key_header)]) -> None:
        # compare_digest takes the same time wherever the first difference is, so timing reveals nothing about
        # how much of a guess was right; == would stop at the first differing byte.
        if provided is None or not hmac.compare_digest(provided.encode(), expected_bytes):
            logger.warning("rejected request without a valid API key", extra={"key_sent": provided is not None})
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=INVALID_API_KEY,
                headers={"WWW-Authenticate": API_KEY_HEADER},
            )

    return check
