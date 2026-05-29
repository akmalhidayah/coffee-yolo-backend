from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.user_service import update_profile

router = APIRouter(prefix="/users", tags=["Users"])


class ProfileRequest(BaseModel):
    name: str = Field(min_length=1)
    email: str = Field(min_length=3)
    location: str = Field(min_length=1)
    phone: str = ""
    auth_provider: str = "email"


@router.post("/profile")
def save_profile(payload: ProfileRequest) -> dict:
    user = update_profile(
        name=payload.name,
        email=payload.email,
        location=payload.location,
        phone=payload.phone,
        auth_provider=payload.auth_provider,
    )
    return {
        "success": True,
        "message": "Profile saved successfully.",
        "data": user,
    }
