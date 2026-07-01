import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import Depends, Header, HTTPException, status

from app.core.config import settings
from app.services.user_service import get_user_by_id, mark_user_seen, serialize_user


def create_access_token(user: Dict[str, Any]) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user["id"]),
        "email": user["email"],
        "name": user["name"],
        "role": user.get("role", "user"),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=settings.jwt_exp_seconds)).timestamp()),
    }
    header = {"alg": settings.jwt_algorithm, "typ": "JWT"}
    signing_input = ".".join(
        [
            _base64_url(json.dumps(header, separators=(",", ":")).encode("utf-8")),
            _base64_url(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
        ]
    )
    signature = hmac.new(
        settings.jwt_secret.encode("utf-8"),
        signing_input.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return f"{signing_input}.{_base64_url(signature)}"


def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        header_part, payload_part, signature_part = token.split(".")
    except ValueError as exc:
        raise ValueError("Token tidak valid.") from exc

    signing_input = f"{header_part}.{payload_part}"
    expected_signature = hmac.new(
        settings.jwt_secret.encode("utf-8"),
        signing_input.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(_base64_url(expected_signature), signature_part):
        raise ValueError("Token tidak valid.")

    header = _json_from_base64(header_part)
    if header.get("alg") != settings.jwt_algorithm:
        raise ValueError("Algoritma token tidak valid.")

    payload = _json_from_base64(payload_part)
    expires_at = payload.get("exp")
    if expires_at is None or int(expires_at) < int(datetime.now(timezone.utc).timestamp()):
        raise ValueError("Token sudah kedaluwarsa.")
    return payload


def authenticate_bearer_token(authorization: Optional[str]) -> Dict[str, Any]:
    token = _extract_bearer_token(authorization)
    payload = decode_access_token(token)
    user = get_user_by_id(str(payload.get("sub", "")))
    if user is None:
        raise ValueError("User token tidak ditemukan.")
    mark_user_seen(str(user["id"]))
    return serialize_user(get_user_by_id(str(user["id"])))


def get_current_user(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    try:
        return authenticate_bearer_token(authorization)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def get_optional_current_user(
    authorization: Optional[str] = Header(default=None),
) -> Optional[Dict[str, Any]]:
    if not authorization:
        return None
    try:
        return authenticate_bearer_token(authorization)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def get_current_admin_user(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Akses ditolak. Hanya admin yang dapat mengakses endpoint ini.",
        )
    return current_user


def _extract_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise ValueError("Token Authorization Bearer wajib dikirim.")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise ValueError("Format Authorization harus Bearer <token>.")
    return token.strip()


def _base64_url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("utf-8").rstrip("=")


def _json_from_base64(value: str) -> Dict[str, Any]:
    padded = value + "=" * (-len(value) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(padded.encode("utf-8")))
    except Exception as exc:
        raise ValueError("Payload token tidak valid.") from exc
