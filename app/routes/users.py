from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.services.token_service import get_current_user
from app.services.user_service import DuplicateEmailError, UserNotFoundError, update_profile

router = APIRouter(prefix="/users", tags=["Users"])


class ProfileRequest(BaseModel):
    name: str = Field(min_length=1)
    email: str = Field(min_length=3)
    location: str = Field(min_length=1)
    phone: str = ""


@router.post("/profile")
def save_profile(
    payload: ProfileRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> dict:
    try:
        user = update_profile(
            name=payload.name,
            email=payload.email,
            location=payload.location,
            phone=payload.phone,
            user_id=current_user["id"],
        )
    except DuplicateEmailError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except UserNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    return {
        "success": True,
        "message": "Profile saved successfully.",
        "data": user,
    }
