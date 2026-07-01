from typing import Any, Dict

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.services.token_service import get_current_user
from app.services.user_service import update_profile

router = APIRouter(prefix="/users", tags=["Users"])


class ProfileRequest(BaseModel):
    name: str = Field(min_length=1)
    email: str = Field(min_length=3)
    location: str = Field(min_length=1)
    phone: str = ""
    auth_provider: str = "email"


@router.post("/profile")
def save_profile(
    payload: ProfileRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> dict:
    user = update_profile(
        name=payload.name,
        email=payload.email,
        location=payload.location,
        phone=payload.phone,
        auth_provider=payload.auth_provider,
        user_id=current_user["id"],
    )
    return {
        "success": True,
        "message": "Profile saved successfully.",
        "data": user,
    }
