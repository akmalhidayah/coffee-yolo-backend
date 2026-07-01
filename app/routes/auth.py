from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.google_auth_service import verify_google_token
from app.services.token_service import create_access_token
from app.services.user_service import login_user, mark_user_login, register_user

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


class GoogleLoginRequest(BaseModel):
    id_token: str = Field(min_length=20)


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
        "access_token": create_access_token(user),
        "token_type": "bearer",
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
        "access_token": create_access_token(user),
        "token_type": "bearer",
    }


@router.post("/google")
def google_login(payload: GoogleLoginRequest) -> dict:
    token_info = verify_google_token(payload.id_token)
    email = token_info["email"].strip().lower()
    name = token_info.get("name") or email.split("@")[0]
    user = register_user(
        name=name,
        email=email,
        password="",
        location="Desa Masewe, Mamasa",
        auth_provider="google",
    )
    user = mark_user_login(user["id"])
    return {
        "success": True,
        "message": "Google login successful.",
        "data": user,
        "access_token": create_access_token(user),
        "token_type": "bearer",
    }
