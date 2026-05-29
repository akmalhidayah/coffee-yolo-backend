from typing import Any, Dict

from fastapi import HTTPException, status
from google.auth.transport import requests
from google.oauth2 import id_token as google_id_token

from app.core.config import settings


def verify_google_token(id_token: str) -> Dict[str, Any]:
    try:
        token_info = google_id_token.verify_oauth2_token(
            id_token,
            requests.Request(),
            settings.google_client_id or None,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token Google tidak valid.",
        ) from exc

    email = token_info.get("email")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token Google tidak memiliki email.",
        )

    if token_info.get("email_verified") is False:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email Google belum terverifikasi.",
        )

    return token_info
