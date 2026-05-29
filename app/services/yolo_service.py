from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from ultralytics import YOLO

from app.core.config import settings


class YoloPredictionError(RuntimeError):
    """Raised when the YOLO model cannot produce a valid prediction."""


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
        results = model.predict(source=str(image), verbose=False)
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
    best_index = _best_detection_index(boxes)
    class_id = int(boxes.cls[best_index].item())
    confidence = round(float(boxes.conf[best_index].item()), 3)

    class_info = _CLASS_BY_ID.get(class_id)
    if class_info is None:
        return _not_detected_response(image.name)

    class_name, coffee_type, grade = class_info
    confidence_percent = round(confidence * 100, 1)
    status = _STATUS_BY_GRADE[grade]
    bounding_boxes = _build_bounding_boxes(
        boxes,
        image_width=image_width,
        image_height=image_height,
    )

    return {
        "image_name": image.name,
        "class_name": class_name,
        "coffee_type": coffee_type,
        "grade": grade,
        "confidence": confidence,
        "confidence_percent": confidence_percent,
        "status": status,
        "description": (
            f"Biji kopi terdeteksi sebagai {class_name} "
            f"dengan {status.lower()}."
        ),
        "recommendation": _RECOMMENDATION_BY_GRADE[grade],
        "characteristics": _CHARACTERISTICS_BY_GRADE[grade],
        "bounding_boxes": bounding_boxes,
    }


def _get_model() -> YOLO:
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    model_path = settings.model_path
    if not model_path.exists():
        raise YoloPredictionError(f"YOLO model file not found: {model_path}")

    try:
        _MODEL = YOLO(str(model_path))
    except Exception as exc:
        raise YoloPredictionError(f"Failed to load YOLO model: {exc}") from exc

    return _MODEL


def _best_detection_index(boxes: Any) -> int:
    confidences = boxes.conf
    if hasattr(confidences, "argmax"):
        return int(confidences.argmax().item())
    return max(range(len(confidences)), key=lambda index: float(confidences[index]))


def _build_bounding_boxes(
    boxes: Any,
    *,
    image_width: int,
    image_height: int,
) -> List[Dict[str, Any]]:
    bounding_boxes = []

    for index in range(len(boxes)):
        class_id = int(boxes.cls[index].item())
        class_info = _CLASS_BY_ID.get(class_id)
        if class_info is None:
            continue

        class_name, coffee_type, grade = class_info
        confidence = round(float(boxes.conf[index].item()), 3)
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
        bounding_boxes.append(bounding_box)

    return bounding_boxes


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
        "description": "Objek biji kopi tidak terdeteksi pada gambar.",
        "recommendation": (
            "Gunakan gambar biji kopi yang lebih jelas dengan pencahayaan cukup."
        ),
        "characteristics": _DEFAULT_CHARACTERISTICS,
        "bounding_boxes": [],
    }
