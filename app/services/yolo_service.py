from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from ultralytics import YOLO

from app.core.config import settings


class YoloPredictionError(RuntimeError):
    """Raised when the YOLO model cannot produce a valid prediction."""


class YoloModelNotAvailableError(YoloPredictionError):
    """Raised when the YOLO model file is not available."""


_MODEL: Optional[YOLO] = None

_CLASS_BY_ID = {
    0: ("Arabica Grade A", "Arabica", "Grade A"),
    1: ("Arabica Grade B", "Arabica", "Grade B"),
    2: ("Arabica Grade C", "Arabica", "Grade C"),
    3: ("Robusta Grade A", "Robusta", "Grade A"),
    4: ("Robusta Grade B", "Robusta", "Grade B"),
    5: ("Robusta Grade C", "Robusta", "Grade C"),
}

_STATUS_BY_GRADE = {
    "Grade A": "Kualitas Tinggi",
    "Grade B": "Kualitas Sedang",
    "Grade C": "Kualitas Rendah",
}

_RECOMMENDATION_BY_GRADE = {
    "Grade A": "Layak jual kualitas tinggi.",
    "Grade B": "Masih layak, perlu sortasi ringan.",
    "Grade C": "Perlu sortasi ulang karena kualitas rendah.",
}

_CHARACTERISTICS_BY_GRADE = {
    "Grade A": {
        "bentuk_keutuhan": "Biji utuh dan bentuk relatif seragam.",
        "ukuran": "Ukuran biji relatif seragam.",
        "permukaan": "Permukaan biji halus dan baik.",
        "warna": "Warna biji merata dan tidak terdapat cacat mencolok.",
    },
    "Grade B": {
        "bentuk_keutuhan": "Sebagian besar biji utuh dengan sedikit variasi bentuk.",
        "ukuran": "Ukuran biji cukup seragam.",
        "permukaan": "Permukaan biji cukup baik dengan sedikit ketidakteraturan.",
        "warna": "Warna biji cukup merata dengan sedikit variasi.",
    },
    "Grade C": {
        "bentuk_keutuhan": "Biji memiliki bentuk kurang seragam dan sebagian tidak utuh.",
        "ukuran": "Ukuran biji tidak seragam.",
        "permukaan": "Permukaan biji kurang halus dan terdapat cacat visual.",
        "warna": "Warna biji tidak merata dan terdapat cacat mencolok.",
    },
}

_DEFAULT_CHARACTERISTICS = {
    "bentuk_keutuhan": "Tidak tersedia.",
    "ukuran": "Tidak tersedia.",
    "permukaan": "Tidak tersedia.",
    "warna": "Tidak tersedia.",
}


def predict_coffee_quality(image_path: str) -> dict:
    """Run YOLO inference and return coffee quality prediction data."""
    image = Path(image_path)
    if not image.exists():
        raise YoloPredictionError(f"Image file not found: {image}")

    try:
        model = _get_model()
        # Threshold ini mengurangi false detection pada background atau objek non-kopi.
        results = model.predict(
            source=str(image),
            conf=settings.confidence_threshold,
            verbose=False,
        )
    except YoloPredictionError:
        raise
    except Exception as exc:
        raise YoloPredictionError(f"Prediction failed: {exc}") from exc

    if not results:
        return _not_detected_response(image.name)

    first_result = results[0]
    boxes = getattr(first_result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return _not_detected_response(image.name)

    image_width, image_height = _get_image_size(image, first_result)
    detections = _build_detections(
        boxes,
        image_width=image_width,
        image_height=image_height,
    )
    if not detections:
        return _not_detected_response(image.name)

    best_detection = max(detections, key=lambda detection: detection["confidence"])
    confidence = best_detection["confidence"]
    confidence_percent = round(confidence * 100, 1)
    grade = best_detection["grade"]
    status = _STATUS_BY_GRADE[grade]
    bounding_boxes = [detection["bounding_box"] for detection in detections]

    return {
        "image_name": image.name,
        "class_name": best_detection["class_name"],
        "coffee_type": best_detection["coffee_type"],
        "grade": grade,
        "confidence": confidence,
        "confidence_percent": confidence_percent,
        "status": status,
        "detection_status": "detected",
        "message": "Biji kopi berhasil terdeteksi.",
        "description": (
            f"Biji kopi terdeteksi sebagai {best_detection['class_name']} "
            f"dengan {status.lower()}."
        ),
        "recommendation": _RECOMMENDATION_BY_GRADE[grade],
        "characteristics": _CHARACTERISTICS_BY_GRADE[grade],
        "bounding_boxes": bounding_boxes,
        "detections": detections,
        "total_detected": len(detections),
        "confidence_threshold": settings.confidence_threshold,
        "detected_at": best_detection["detected_at"],
    }


def _get_model() -> YOLO:
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    model_path = settings.model_path
    if not model_path.exists():
        raise YoloModelNotAvailableError(f"YOLO model file not found: {model_path}")

    try:
        _MODEL = YOLO(str(model_path))
    except Exception as exc:
        raise YoloModelNotAvailableError(f"Failed to load YOLO model: {exc}") from exc

    return _MODEL


def reset_model_cache() -> None:
    global _MODEL
    _MODEL = None


def _build_detections(
    boxes: Any,
    *,
    image_width: int,
    image_height: int,
) -> List[Dict[str, Any]]:
    detections = []

    for index in range(len(boxes)):
        confidence = round(float(boxes.conf[index].item()), 3)
        if confidence < settings.confidence_threshold:
            continue

        class_id = int(boxes.cls[index].item())
        class_info = _CLASS_BY_ID.get(class_id)
        if class_info is None:
            continue

        class_name, coffee_type, grade = class_info
        bounding_box = _normalize_box(
            boxes.xyxy[index].tolist(),
            image_width=image_width,
            image_height=image_height,
            confidence=confidence,
        )
        bounding_box["label"] = class_name
        bounding_box["class_name"] = class_name
        bounding_box["coffee_type"] = coffee_type
        bounding_box["grade"] = grade
        detected_at = datetime.now(timezone.utc).isoformat()
        detections.append(
            {
                "label": class_name,
                "class_name": class_name,
                "coffee_type": coffee_type,
                "jenis_kopi": coffee_type,
                "grade": grade,
                "confidence": confidence,
                "confidence_percent": round(confidence * 100, 1),
                "bbox": bounding_box,
                "bounding_box": bounding_box,
                "recommendation": _RECOMMENDATION_BY_GRADE[grade],
                "rekomendasi": _RECOMMENDATION_BY_GRADE[grade],
                "characteristics": _CHARACTERISTICS_BY_GRADE[grade],
                "karakteristik": _CHARACTERISTICS_BY_GRADE[grade],
                "detected_at": detected_at,
            }
        )

    return detections


def _get_image_size(image: Path, result: Any) -> Tuple[int, int]:
    orig_shape = getattr(result, "orig_shape", None)
    if orig_shape and len(orig_shape) >= 2:
        height, width = int(orig_shape[0]), int(orig_shape[1])
        return width, height

    with Image.open(image) as uploaded_image:
        return uploaded_image.size


def _normalize_box(
    xyxy: List[float],
    *,
    image_width: int,
    image_height: int,
    confidence: float,
) -> dict:
    x1, y1, x2, y2 = xyxy
    width = max(image_width, 1)
    height = max(image_height, 1)

    return {
        "x": _clamp(round(x1 / width, 4)),
        "y": _clamp(round(y1 / height, 4)),
        "width": _clamp(round((x2 - x1) / width, 4)),
        "height": _clamp(round((y2 - y1) / height, 4)),
        "confidence": confidence,
    }


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _not_detected_response(image_name: str) -> dict:
    return {
        "image_name": image_name,
        "class_name": "Tidak Terdeteksi",
        "coffee_type": "-",
        "grade": "-",
        "confidence": 0,
        "confidence_percent": 0,
        "status": "Tidak Terdeteksi",
        "detection_status": "not_detected",
        "message": (
            "Tidak ada biji kopi terdeteksi. Silakan ambil gambar ulang dengan "
            "pencahayaan yang cukup dan objek biji kopi terlihat jelas."
        ),
        "description": "Objek biji kopi tidak terdeteksi pada gambar.",
        "recommendation": (
            "Gunakan gambar biji kopi yang lebih jelas dengan pencahayaan cukup."
        ),
        "characteristics": _DEFAULT_CHARACTERISTICS,
        "bounding_boxes": [],
        "detections": [],
        "total_detected": 0,
        "confidence_threshold": settings.confidence_threshold,
    }
