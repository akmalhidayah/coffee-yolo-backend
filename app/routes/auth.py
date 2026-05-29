from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.user_service import login_user, register_user

router = APIRouter(prefix="/auth", tags=["Authentication"])


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1)
    email: str = Field(min_length=3)
    password: str = ""
    location: str = "Desa Masewe, Mamasa"
    phone: str = ""
    auth_provider: str = "email"


class LoginRequest(BaseModel):
    email: str = Field(min_length=3)
    password: str = ""


@router.post("/register")
def register(payload: RegisterRequest) -> dict:
    user = register_user(
        name=payload.name,
        email=payload.email,
        password=payload.password,
        location=payload.location,
        phone=payload.phone,
        auth_provider=payload.auth_provider,
    )
    return {
        "success": True,
        "message": "Account saved successfully.",
        "data": user,
    }


@router.post("/login")
def login(payload: LoginRequest) -> dict:
    user = login_user(email=payload.email, password=payload.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email atau password tidak sesuai.",
        )

    return {
        "success": True,
        "message": "Login successful.",
        "data": user,
    }
