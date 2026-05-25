from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.services.yolo_service import YoloPredictionError, predict_coffee_quality

router = APIRouter(tags=["Prediction"])


@router.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    extension = _get_extension(file.filename)
    if extension not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only jpg, jpeg, and png are allowed.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    unique_name = f"{uuid4().hex}.{extension}"
    image_path = settings.upload_dir / unique_name
    image_path.write_bytes(contents)

    try:
        prediction = predict_coffee_quality(str(image_path))
    except YoloPredictionError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    prediction["image_name"] = unique_name

    return {
        "success": True,
        "message": "Prediction completed",
        "data": prediction,
    }


def _get_extension(filename: str | None) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower().lstrip(".")
