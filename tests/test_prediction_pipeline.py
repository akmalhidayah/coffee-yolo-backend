from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import UploadFile
from PIL import Image

from app.core.config import settings
from app.routes.predict import _is_valid_image_upload
from app.services import yolo_service


class EmptyBoxes:
    def __len__(self):
        return 0


class FakeModel:
    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        source = kwargs["source"]
        return [SimpleNamespace(boxes=EmptyBoxes(), orig_shape=(source.height, source.width))]


def _image_bytes(format_name="JPEG", size=(12, 8)):
    output = BytesIO()
    Image.new("RGB", size, color=(90, 60, 30)).save(output, format=format_name)
    return output.getvalue()


def test_prepare_inference_image_transposes_exif_to_rgb_without_rewriting(tmp_path):
    path = tmp_path / "oriented.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("L", (10, 20), color=100).save(path, exif=exif)
    original = path.read_bytes()

    prepared, info = yolo_service.prepare_inference_image(path)
    try:
        assert prepared.mode == "RGB"
        assert prepared.size == (20, 10)
        assert info["original_width"] == 10
        assert info["original_height"] == 20
        assert info["exif_orientation"] == 6
        assert info["exif_transposed"] is True
    finally:
        prepared.close()
    assert path.read_bytes() == original


def test_prepare_inference_image_without_exif_is_not_rotated(tmp_path):
    path = tmp_path / "plain.png"
    path.write_bytes(_image_bytes("PNG", (13, 7)))
    original = path.read_bytes()

    prepared, info = yolo_service.prepare_inference_image(path)
    try:
        assert prepared.mode == "RGB"
        assert prepared.size == (13, 7)
        assert info["exif_transposed"] is False
    finally:
        prepared.close()
    assert path.read_bytes() == original


def test_predict_uses_normalized_memory_image_and_explicit_parameters(
    tmp_path, monkeypatch
):
    path = tmp_path / "input.png"
    path.write_bytes(_image_bytes("PNG"))
    model = FakeModel()
    monkeypatch.setattr(yolo_service, "_get_model", lambda: model)

    result = yolo_service.predict_coffee_quality(str(path))

    call = model.calls[0]
    assert isinstance(call["source"], Image.Image)
    assert call["conf"] == pytest.approx(0.5)
    assert call["imgsz"] == 640
    assert call["iou"] == pytest.approx(0.70)
    assert call["max_det"] == 300
    assert call["device"] == "cpu"
    assert call["half"] is False
    assert call["augment"] is False
    assert call["verbose"] is False
    assert result["detection_status"] == "not_detected"
    assert result["summary"]["total"] == 0
    assert result["input_info"]["file_size_bytes"] == path.stat().st_size
    assert result["inference_parameters"]["image_size"] == 640
    for old_field in (
        "class_name",
        "coffee_type",
        "grade",
        "confidence",
        "confidence_percent",
        "status",
        "description",
        "recommendation",
        "characteristics",
        "bounding_boxes",
        "detections",
        "total_detected",
        "summary",
        "aggregation_method",
    ):
        assert old_field in result


def test_normalized_bounding_box_remains_between_zero_and_one():
    box = yolo_service._normalize_box(
        [-20, 10, 1200, 800], image_width=1000, image_height=500, confidence=0.8
    )
    assert 0 <= box["x"] <= 1
    assert 0 <= box["y"] <= 1
    assert 0 <= box["width"] <= 1
    assert 0 <= box["height"] <= 1
    assert box["x"] + box["width"] <= 1
    assert box["y"] + box["height"] <= 1


@pytest.mark.parametrize(
    ("filename", "mime", "format_name"),
    [("coffee.jpg", "image/jpeg", "JPEG"), ("coffee.png", "image/png", "PNG")],
)
def test_valid_jpg_and_png_uploads_are_accepted(filename, mime, format_name):
    contents = _image_bytes(format_name)
    upload = UploadFile(filename=filename, file=BytesIO(contents), headers={"content-type": mime})
    assert _is_valid_image_upload(upload, contents, filename.rsplit(".", 1)[1])


def test_empty_and_oversized_uploads_are_rejected():
    upload = UploadFile(filename="coffee.jpg", file=BytesIO(), headers={"content-type": "image/jpeg"})
    assert not _is_valid_image_upload(upload, b"", "jpg")
    oversized = b"x" * (settings.max_image_size_bytes + 1)
    assert not _is_valid_image_upload(upload, oversized, "jpg")


def test_internal_exception_is_wrapped_and_sensitive_values_are_not_logged(
    tmp_path, monkeypatch, caplog
):
    path = tmp_path / "safe-name.png"
    path.write_bytes(_image_bytes("PNG"))

    class BrokenModel:
        def predict(self, **kwargs):
            raise RuntimeError("token=secret email=user@example.com password=hunter2")

    monkeypatch.setattr(yolo_service, "_get_model", lambda: BrokenModel())
    with pytest.raises(yolo_service.YoloPredictionError) as caught:
        yolo_service.predict_coffee_quality(str(path))
    assert str(caught.value) == "Prediksi tidak dapat diproses."
    # The summary message contains only the internal UUID/name, never auth data.
    record_messages = [record.getMessage() for record in caplog.records]
    assert any("Prediction failed image=safe-name.png" in item for item in record_messages)
    assert all("token=" not in item for item in record_messages)
    assert all("user@example.com" not in item for item in record_messages)
    assert "token=secret" not in caplog.text
    assert "user@example.com" not in caplog.text
    assert "hunter2" not in caplog.text
