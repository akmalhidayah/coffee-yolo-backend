import os
import shutil
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image
from ultralytics import YOLO

from app.core.config import settings


class YoloPredictionError(RuntimeError):
    """Raised when YOLO cannot produce a valid prediction."""


class YoloModelNotAvailableError(YoloPredictionError):
    """Raised when the active model is unavailable."""


class InvalidYoloModelError(RuntimeError):
    """Raised when an uploaded checkpoint is incompatible with this system."""


EXPECTED_MODEL_NAMES = {
    0: "Arabica Grade A",
    1: "Arabica Grade B",
    2: "Arabica Grade C",
    3: "Robusta Grade A",
    4: "Robusta Grade B",
    5: "Robusta Grade C",
}

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
_DEFAULT_CHARACTERISTICS = {key: "Tidak tersedia." for key in (
    "bentuk_keutuhan", "ukuran", "permukaan", "warna"
)}

CHARACTERISTICS_SOURCE = "rule_based_from_dominant_grade"
CHARACTERISTICS_NOTE = (
    "Karakteristik berikut merupakan karakteristik umum yang diasosiasikan "
    "dengan grade dominan berdasarkan aturan sistem, bukan hasil pengukuran "
    "warna, ukuran, permukaan, atau keutuhan secara terpisah oleh model YOLO."
)
RECOMMENDATION_SOURCE = "rule_based_from_dominant_grade"
RECOMMENDATION_NOTE = (
    "Rekomendasi dibuat oleh aturan sistem berdasarkan grade dominan hasil deteksi."
)

_MODEL: Optional[YOLO] = None
_MODEL_LOCK = threading.RLock()
_MODEL_UPLOAD_LOCK = threading.Lock()


def predict_coffee_quality(image_path: str) -> Dict[str, Any]:
    image = Path(image_path)
    if not image.exists():
        raise YoloPredictionError("File gambar tidak ditemukan.")
    try:
        model = _get_model()
        results = model.predict(
            source=str(image), conf=settings.confidence_threshold, verbose=False
        )
    except YoloPredictionError:
        raise
    except Exception as exc:
        raise YoloPredictionError("Prediksi tidak dapat diproses.") from exc

    if not results or getattr(results[0], "boxes", None) is None:
        return _not_detected_response(image.name)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return _not_detected_response(image.name)

    width, height = _get_image_size(image, results[0])
    detected_at = datetime.now(timezone.utc).isoformat()
    detections = _build_detections(
        boxes, image_width=width, image_height=height, detected_at=detected_at
    )
    if not detections:
        return _not_detected_response(image.name)

    aggregate = aggregate_detections(detections)
    grade = aggregate["grade"]
    dominant_count = aggregate["summary"]["dominant_count"]
    total = aggregate["summary"]["total"]
    return {
        "image_name": image.name,
        **aggregate,
        "status": _STATUS_BY_GRADE[grade],
        "detection_status": "detected",
        "message": "Biji kopi berhasil terdeteksi.",
        "description": (
            f"Berdasarkan {total} objek yang terdeteksi, kelas yang paling dominan "
            f"adalah {aggregate['class_name']} sebanyak {dominant_count} objek. "
            "Penetapan hasil keseluruhan menggunakan jumlah objek terbanyak, "
            "bukan hanya satu detection dengan confidence tertinggi."
        ),
        "recommendation": _RECOMMENDATION_BY_GRADE[grade],
        "recommendation_source": RECOMMENDATION_SOURCE,
        "recommendation_note": RECOMMENDATION_NOTE,
        "characteristics": _CHARACTERISTICS_BY_GRADE[grade],
        "characteristics_source": CHARACTERISTICS_SOURCE,
        "characteristics_note": CHARACTERISTICS_NOTE,
        "bounding_boxes": [item["bounding_box"] for item in detections],
        "detections": detections,
        "total_detected": total,
        "confidence_threshold": settings.confidence_threshold,
        "detected_at": detected_at,
        "aggregation_method": "majority_count_then_confidence",
    }


def aggregate_detections(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate detections deterministically without invoking YOLO."""
    class_counts = {name: 0 for name in EXPECTED_MODEL_NAMES.values()}
    grade_counts = {grade: 0 for grade in ("Grade A", "Grade B", "Grade C")}
    coffee_counts = {name: 0 for name in ("Arabica", "Robusta")}
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)

    for detection in detections:
        class_name = str(detection.get("class_name", ""))
        grade = str(detection.get("grade", ""))
        coffee_type = str(detection.get("coffee_type", ""))
        if class_name not in class_counts:
            continue
        class_counts[class_name] += 1
        if grade in grade_counts:
            grade_counts[grade] += 1
        if coffee_type in coffee_counts:
            coffee_counts[coffee_type] += 1
        grouped[(class_name, grade, coffee_type)].append(detection)

    total = sum(class_counts.values())
    if total == 0 or not grouped:
        return {
            "class_name": "-", "coffee_type": "-", "grade": "-",
            "confidence": 0.0, "confidence_percent": 0.0,
            "summary": _empty_summary(),
        }

    def rank(item: Tuple[Tuple[str, str, str], List[Dict[str, Any]]]) -> Tuple[Any, ...]:
        _, values = item
        confidences = [float(value.get("confidence", 0.0)) for value in values]
        class_id = min(int(value.get("class_id", 999)) for value in values)
        total_confidence = sum(confidences)
        average = total_confidence / len(values)
        return (-len(values), -total_confidence, -average, class_id)

    (class_name, grade, coffee_type), dominant = min(grouped.items(), key=rank)
    dominant_confidence = round(
        sum(float(item.get("confidence", 0.0)) for item in dominant) / len(dominant), 3
    )
    valid_detections = [item for values in grouped.values() for item in values]
    average_confidence = round(
        sum(float(item.get("confidence", 0.0)) for item in valid_detections) / total, 3
    )
    dominant_count = len(dominant)
    summary = {
        "total": total,
        "class_counts": class_counts,
        "grade_counts": grade_counts,
        "coffee_type_counts": coffee_counts,
        "dominant_class": class_name,
        "dominant_coffee_type": coffee_type,
        "dominant_grade": grade,
        "dominant_count": dominant_count,
        "dominant_percentage": round(dominant_count / total * 100, 1),
        "dominant_average_confidence": dominant_confidence,
        "average_confidence": average_confidence,
        "low_quality_percentage": round(grade_counts["Grade C"] / total * 100, 1),
        "sorting_required_percentage": round(
            (grade_counts["Grade B"] + grade_counts["Grade C"]) / total * 100, 1
        ),
    }
    return {
        "class_name": class_name,
        "coffee_type": coffee_type,
        "grade": grade,
        "confidence": dominant_confidence,
        "confidence_percent": round(dominant_confidence * 100, 1),
        "summary": summary,
    }


def normalize_model_names(names: Any) -> Dict[int, str]:
    if isinstance(names, Mapping):
        try:
            return {int(index): str(name) for index, name in names.items()}
        except (TypeError, ValueError) as exc:
            raise InvalidYoloModelError("Nama kelas model tidak valid.") from exc
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return {index: str(name) for index, name in enumerate(names)}
    raise InvalidYoloModelError("Nama kelas model tidak valid.")


def validate_candidate_model(path: Path) -> Tuple[YOLO, Dict[int, str]]:
    try:
        model = YOLO(str(path))
    except Exception as exc:
        raise InvalidYoloModelError("Model tidak valid atau tidak dapat dimuat.") from exc
    if getattr(model, "task", None) != "detect":
        raise InvalidYoloModelError("Model harus menggunakan task detection.")
    names = normalize_model_names(getattr(model, "names", None))
    if len(names) != 6 or set(names) != set(range(6)):
        raise InvalidYoloModelError("Model harus memiliki tepat 6 kelas.")
    if names != EXPECTED_MODEL_NAMES:
        raise InvalidYoloModelError("Urutan nama kelas model tidak sesuai dengan sistem.")

    smoke_path: Optional[Path] = None
    try:
        with NamedTemporaryFile(
            suffix=".jpg", dir=path.parent, delete=False
        ) as smoke_file:
            smoke_path = Path(smoke_file.name)
        Image.new("RGB", (640, 640), color=(128, 128, 128)).save(smoke_path)
        results = model.predict(source=str(smoke_path), verbose=False)
        if not isinstance(results, list) or not results or not hasattr(results[0], "boxes"):
            raise InvalidYoloModelError("Model gagal menjalankan smoke inference.")
    except InvalidYoloModelError:
        raise
    except Exception as exc:
        raise InvalidYoloModelError("Model gagal menjalankan smoke inference.") from exc
    finally:
        if smoke_path is not None:
            smoke_path.unlink(missing_ok=True)
    return model, names


def validate_and_activate_model(candidate_path: Path) -> Dict[str, Any]:
    """Validate then atomically activate one upload at a time."""
    global _MODEL
    with _MODEL_UPLOAD_LOCK:
        model, names = validate_candidate_model(candidate_path)
        settings.model_path.parent.mkdir(parents=True, exist_ok=True)
        if settings.model_path.exists():
            settings.model_backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(settings.model_path, settings.model_backup_path)
        with _MODEL_LOCK:
            os.replace(candidate_path, settings.model_path)
            _MODEL = model
        return {"task": "detect", "class_count": 6, "class_names": names}


def _get_model() -> YOLO:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        if not settings.model_path.exists():
            raise YoloModelNotAvailableError("Model deteksi belum tersedia.")
        try:
            _MODEL = YOLO(str(settings.model_path))
        except Exception as exc:
            raise YoloModelNotAvailableError("Model deteksi tidak dapat dimuat.") from exc
        return _MODEL


def reset_model_cache() -> None:
    global _MODEL
    with _MODEL_LOCK:
        _MODEL = None


def _build_detections(
    boxes: Any, *, image_width: int, image_height: int, detected_at: str
) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    for index in range(len(boxes)):
        confidence = round(float(boxes.conf[index].item()), 3)
        if confidence < settings.confidence_threshold:
            continue
        class_id = int(boxes.cls[index].item())
        class_info = _CLASS_BY_ID.get(class_id)
        if class_info is None:
            continue
        class_name, coffee_type, grade = class_info
        bbox = _normalize_box(
            boxes.xyxy[index].tolist(), image_width=image_width,
            image_height=image_height, confidence=confidence
        )
        bbox.update({"label": class_name, "class_name": class_name,
                     "coffee_type": coffee_type, "grade": grade})
        detections.append({
            "class_id": class_id, "label": class_name, "class_name": class_name,
            "coffee_type": coffee_type, "jenis_kopi": coffee_type, "grade": grade,
            "confidence": confidence, "confidence_percent": round(confidence * 100, 1),
            "bbox": bbox, "bounding_box": bbox,
            "recommendation": _RECOMMENDATION_BY_GRADE[grade],
            "rekomendasi": _RECOMMENDATION_BY_GRADE[grade],
            "characteristics": _CHARACTERISTICS_BY_GRADE[grade],
            "karakteristik": _CHARACTERISTICS_BY_GRADE[grade],
            "detected_at": detected_at,
        })
    return detections


def _empty_summary() -> Dict[str, Any]:
    return {
        "total": 0,
        "class_counts": {name: 0 for name in EXPECTED_MODEL_NAMES.values()},
        "grade_counts": {name: 0 for name in ("Grade A", "Grade B", "Grade C")},
        "coffee_type_counts": {name: 0 for name in ("Arabica", "Robusta")},
        "dominant_class": "-", "dominant_coffee_type": "-", "dominant_grade": "-",
        "dominant_count": 0, "dominant_percentage": 0.0,
        "dominant_average_confidence": 0.0, "average_confidence": 0.0,
        "low_quality_percentage": 0.0, "sorting_required_percentage": 0.0,
    }


def _not_detected_response(image_name: str) -> Dict[str, Any]:
    return {
        "image_name": image_name, "class_name": "Tidak Terdeteksi",
        "coffee_type": "-", "grade": "-", "confidence": 0.0,
        "confidence_percent": 0.0, "status": "Tidak Terdeteksi",
        "detection_status": "not_detected",
        "message": "Tidak ada biji kopi terdeteksi. Silakan ambil gambar ulang dengan pencahayaan yang cukup dan objek biji kopi terlihat jelas.",
        "description": "Objek biji kopi tidak terdeteksi pada gambar.",
        "recommendation": "Gunakan gambar biji kopi yang lebih jelas dengan pencahayaan cukup.",
        "recommendation_source": RECOMMENDATION_SOURCE,
        "recommendation_note": RECOMMENDATION_NOTE,
        "characteristics": _DEFAULT_CHARACTERISTICS,
        "characteristics_source": CHARACTERISTICS_SOURCE,
        "characteristics_note": CHARACTERISTICS_NOTE,
        "bounding_boxes": [], "detections": [], "total_detected": 0,
        "confidence_threshold": settings.confidence_threshold,
        "aggregation_method": "majority_count_then_confidence",
        "summary": _empty_summary(),
    }


def _get_image_size(image: Path, result: Any) -> Tuple[int, int]:
    shape = getattr(result, "orig_shape", None)
    if shape and len(shape) >= 2:
        return int(shape[1]), int(shape[0])
    with Image.open(image) as uploaded:
        return uploaded.size


def _normalize_box(
    xyxy: List[float], *, image_width: int, image_height: int, confidence: float
) -> Dict[str, float]:
    x1, y1, x2, y2 = xyxy
    width, height = max(image_width, 1), max(image_height, 1)
    x, y = _clamp(round(x1 / width, 4)), _clamp(round(y1 / height, 4))
    box_width = min(_clamp(round((x2 - x1) / width, 4)), 1.0 - x)
    box_height = min(_clamp(round((y2 - y1) / height, 4)), 1.0 - y)
    return {"x": x, "y": y, "width": box_width, "height": box_height,
            "confidence": confidence}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
