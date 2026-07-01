from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from fastapi import APIRouter, File, Header, UploadFile, status
from fastapi.responses import JSONResponse
from ultralytics import YOLO

from app.core.config import settings
from app.services.token_service import authenticate_bearer_token
from app.services.yolo_service import reset_model_cache

router = APIRouter(prefix="/models", tags=["Model Management"])


@router.post("/upload")
async def upload_model(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
    x_admin_token: Optional[str] = Header(default=None),
) -> dict:
    admin_user, auth_error = _resolve_upload_admin(authorization, x_admin_token)
    if auth_error is not None:
        return auth_error

    extension = _get_extension(file.filename)
    if extension not in settings.allowed_model_extensions:
        return _error_response(
            "Invalid model file. Only .pt files are allowed.",
            http_status=status.HTTP_400_BAD_REQUEST,
        )

    contents = await file.read()
    if not contents:
        return _error_response(
            "Uploaded model file is empty.",
            http_status=status.HTTP_400_BAD_REQUEST,
        )

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = settings.model_path.parent / f"upload-{uuid4().hex}.pt"
    temp_path.write_bytes(contents)

    try:
        validated_model = YOLO(str(temp_path))
        del validated_model
    except Exception:
        temp_path.unlink(missing_ok=True)
        return _error_response(
            "Model tidak valid atau tidak dapat diproses.",
            http_status=status.HTTP_400_BAD_REQUEST,
        )

    if settings.model_path.exists():
        settings.model_backup_path.write_bytes(settings.model_path.read_bytes())

    settings.model_path.write_bytes(temp_path.read_bytes())
    temp_path.unlink(missing_ok=True)
    reset_model_cache()

    return {
        "success": True,
        "status": "success",
        "message": "Model uploaded successfully.",
        "data": {
            "model_name": settings.model_path.name,
            "size_bytes": len(contents),
            "uploaded_by": admin_user.get("email") if admin_user else "fallback-token",
            "uploaded_by_user_id": admin_user.get("id") if admin_user else None,
        },
    }


def _resolve_upload_admin(
    authorization: Optional[str],
    x_admin_token: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[JSONResponse]]:
    if authorization:
        try:
            user = authenticate_bearer_token(authorization)
        except ValueError:
            return None, _error_response(
                "Token tidak valid.",
                http_status=status.HTTP_401_UNAUTHORIZED,
            )
        if user.get("role") != "admin":
            return None, _error_response(
                "Akses ditolak. Hanya admin yang dapat mengunggah model.",
                http_status=status.HTTP_403_FORBIDDEN,
            )
        return user, None

    if settings.model_upload_token and x_admin_token == settings.model_upload_token:
        return None, None

    return None, _error_response(
        "Token Authorization Bearer wajib dikirim.",
        http_status=status.HTTP_401_UNAUTHORIZED,
    )


def _error_response(message: str, *, http_status: int) -> JSONResponse:
    return JSONResponse(
        status_code=http_status,
        content={
            "success": False,
            "status": "error",
            "message": message,
            "detections": [],
            "total_detected": 0,
        },
    )


def _get_extension(filename: Optional[str]) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower().lstrip(".")
