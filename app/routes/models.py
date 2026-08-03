import asyncio
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.services.token_service import get_current_admin_user
from app.services.yolo_service import InvalidYoloModelError, validate_and_activate_model

router = APIRouter(prefix="/models", tags=["Model Management"])
_CHUNK_SIZE = 1024 * 1024
_UPLOAD_REQUEST_LOCK = asyncio.Lock()


@router.post("/upload")
async def upload_model(
    file: UploadFile = File(...),
    admin_user: Dict[str, Any] = Depends(get_current_admin_user),
) -> Any:
    async with _UPLOAD_REQUEST_LOCK:
        return await _process_upload(file, admin_user)


async def _process_upload(
    file: UploadFile,
    admin_user: Dict[str, Any],
) -> Any:
    if _get_extension(file.filename) not in settings.allowed_model_extensions:
        return _error_response(
            "File model harus berformat .pt.", status.HTTP_400_BAD_REQUEST
        )

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = settings.model_path.parent / f"upload-{uuid4().hex}.pt"
    size_bytes = 0
    digest = hashlib.sha256()
    try:
        with temp_path.open("xb") as destination:
            while chunk := await file.read(_CHUNK_SIZE):
                size_bytes += len(chunk)
                if size_bytes > settings.max_model_size_bytes:
                    return _error_response(
                        f"Ukuran model melebihi batas {settings.max_model_size_mb} MB.",
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    )
                digest.update(chunk)
                destination.write(chunk)
        if size_bytes == 0:
            return _error_response("File model kosong.", status.HTTP_400_BAD_REQUEST)
        metadata = await run_in_threadpool(validate_and_activate_model, temp_path)
        return {
            "success": True,
            "status": "success",
            "message": "Model berhasil divalidasi dan diaktifkan.",
            "data": {
                "model_name": settings.model_path.name,
                "size_bytes": size_bytes,
                "sha256": digest.hexdigest(),
                **metadata,
                "uploaded_by": admin_user["email"],
                "uploaded_by_user_id": admin_user["id"],
            },
        }
    except InvalidYoloModelError as exc:
        return _error_response(str(exc), status.HTTP_400_BAD_REQUEST)
    except FileExistsError:
        return _error_response(
            "Upload model tidak dapat disiapkan.", status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    except Exception:
        return _error_response(
            "Model gagal diproses. Model aktif sebelumnya tetap digunakan.",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    finally:
        await file.close()
        temp_path.unlink(missing_ok=True)


def _error_response(message: str, http_status: int) -> JSONResponse:
    return JSONResponse(
        status_code=http_status,
        content={"success": False, "status": "error", "message": message,
                 "detections": [], "total_detected": 0},
    )


def _get_extension(filename: Optional[str]) -> str:
    return Path(filename or "").suffix.lower().lstrip(".")
