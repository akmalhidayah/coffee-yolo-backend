from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.google_auth_service import verify_google_token
from app.services.token_service import create_access_token
from app.services.user_service import (
    DuplicateEmailError,
    InactiveAccountError,
    get_or_create_google_user,
    login_user,
    mark_user_login,
    register_user,
)

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
    try:
        user = register_user(
            name=payload.name,
            email=payload.email,
            password=payload.password,
            location=payload.location,
            phone=payload.phone,
            auth_provider="email",
        )
    except DuplicateEmailError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    return {
        "success": True,
        "message": "Account saved successfully.",
        "data": user,
        "access_token": create_access_token(user),
        "token_type": "bearer",
    }


@router.post("/login")
def login(payload: LoginRequest) -> dict:
    try:
        user = login_user(email=payload.email, password=payload.password)
    except InactiveAccountError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)
        ) from exc
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
    try:
        user = get_or_create_google_user(name=name, email=email)
        user = mark_user_login(user["id"])
    except InactiveAccountError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)
        ) from exc
    return {
        "success": True,
        "message": "Google login successful.",
        "data": user,
        "access_token": create_access_token(user),
        "token_type": "bearer",
    }
