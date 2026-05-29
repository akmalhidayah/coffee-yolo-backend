from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Header, HTTPException, UploadFile, status

from app.core.config import settings
from app.services.yolo_service import reset_model_cache

router = APIRouter(prefix="/models", tags=["Model Management"])


@router.post("/upload")
async def upload_model(
    file: UploadFile = File(...),
    x_admin_token: Optional[str] = Header(default=None),
) -> dict:
    if x_admin_token != settings.model_upload_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token.",
        )

    extension = _get_extension(file.filename)
    if extension not in settings.allowed_model_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid model file. Only .pt files are allowed.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded model file is empty.",
        )

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    if settings.model_path.exists():
        settings.model_backup_path.write_bytes(settings.model_path.read_bytes())

    settings.model_path.write_bytes(contents)
    reset_model_cache()

    return {
        "success": True,
        "message": "Model uploaded successfully.",
        "data": {
            "model_name": settings.model_path.name,
            "size_bytes": len(contents),
        },
    }


def _get_extension(filename: Optional[str]) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower().lstrip(".")
