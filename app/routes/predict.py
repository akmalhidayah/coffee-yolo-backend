from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.services.token_service import get_optional_current_user
from app.services.user_service import record_prediction_history
from app.services.yolo_service import (
    YoloModelNotAvailableError,
    YoloPredictionError,
    predict_coffee_quality,
)

router = APIRouter(tags=["Prediction"])

_ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
) -> dict:
    contents = await file.read()
    extension = _get_extension(file.filename)
    if not _is_valid_image_upload(file, contents, extension):
        return _error_response(
            "File gambar tidak valid atau tidak dapat diproses.",
            http_status=status.HTTP_400_BAD_REQUEST,
        )

    unique_name = f"{uuid4().hex}.{extension}"
    image_path = settings.upload_dir / unique_name
    image_path.write_bytes(contents)

    try:
        prediction = predict_coffee_quality(str(image_path))
    except YoloModelNotAvailableError:
        return _error_response(
            "Model deteksi belum tersedia.",
            http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    except YoloPredictionError:
        return _error_response(
            "Terjadi kesalahan saat memproses gambar.",
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    prediction["image_name"] = unique_name
    response_status = prediction.get("detection_status", "not_detected")
    message = prediction.get("message") or (
        "Biji kopi berhasil terdeteksi."
        if response_status == "detected"
        else "Tidak ada biji kopi terdeteksi."
    )
    record_prediction_history(
        user_id=current_user["id"] if current_user else None,
        image_filename=unique_name,
        response_status=response_status,
        prediction=prediction,
    )

    return {
        "success": True,
        "status": response_status,
        "message": message,
        "detections": prediction.get("detections", []),
        "total_detected": prediction.get("total_detected", 0),
        "confidence_threshold": settings.confidence_threshold,
        "summary": prediction.get("summary", {}),
        "input_info": prediction.get("input_info", {}),
        "inference_parameters": prediction.get("inference_parameters", {}),
        "data": prediction,
    }


def _is_valid_image_upload(file: UploadFile, contents: bytes, extension: str) -> bool:
    if extension not in settings.allowed_extensions:
        return False
    if file.content_type not in _ALLOWED_MIME_TYPES:
        return False
    if not contents or len(contents) > settings.max_image_size_bytes:
        return False
    try:
        with Image.open(BytesIO(contents)) as image:
            image.verify()
            return image.format in {"JPEG", "PNG"}
    except (UnidentifiedImageError, OSError, ValueError):
        return False


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
