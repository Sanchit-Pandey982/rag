"""Phase 8: per-user token usage + cost totals.

Authenticated users see only their own ledger rows; the service layer
scopes every query by the JWT identity, so ids can never be probed.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.dependencies.auth import get_current_user
from app.schemas.auth import UserResponse
from app.schemas.usage import UsageSummaryResponse
from app.services.usage_service import UsageService, UsageStoreUnavailable


router = APIRouter(prefix="/api/v1/usage", tags=["usage"])


def _service(request: Request) -> UsageService:
    service = getattr(request.app.state, "usage_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Usage service is unavailable. Please retry.",
        )
    return service


@router.get("", response_model=UsageSummaryResponse)
async def get_usage(
    request: Request,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
    days: Annotated[int, Query(ge=1, le=31)] = 7,
) -> UsageSummaryResponse:
    """Trailing-window daily totals; ``days=7`` is the weekly view."""
    try:
        summary = await _service(request).get_usage_summary(
            current_user.user_id, days=days
        )
    except UsageStoreUnavailable as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Usage service is unavailable. Please retry.",
        ) from error
    return UsageSummaryResponse(**summary)
