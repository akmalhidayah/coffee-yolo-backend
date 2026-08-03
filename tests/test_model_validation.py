from types import SimpleNamespace

import pytest

from app.services import yolo_service
from app.services.yolo_service import InvalidYoloModelError, normalize_model_names


def test_model_names_accept_exact_six_class_mapping():
    assert normalize_model_names({0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F"})[5] == "F"


def test_model_names_reject_non_numeric_indexes():
    with pytest.raises(InvalidYoloModelError):
        normalize_model_names({"invalid": "Arabica Grade A"})


class FakeModel:
    def __init__(self, *, task="detect", names=None, result=None):
        self.task = task
        self.names = names if names is not None else yolo_service.EXPECTED_MODEL_NAMES
        self.result = result if result is not None else [SimpleNamespace(boxes=[])]

    def predict(self, **_kwargs):
        return self.result


@pytest.mark.parametrize(
    "model, message",
    [
        (FakeModel(task="classify"), "task detection"),
        (FakeModel(names={0: "A"}), "tepat 6 kelas"),
        (FakeModel(names={**yolo_service.EXPECTED_MODEL_NAMES, 0: "Salah"}), "Urutan nama"),
        (FakeModel(names=list(reversed(yolo_service.EXPECTED_MODEL_NAMES.values()))), "Urutan nama"),
        (FakeModel(result=[SimpleNamespace()]), "smoke inference"),
    ],
)
def test_candidate_validation_rejects_incompatible_model(tmp_path, monkeypatch, model, message):
    monkeypatch.setattr(yolo_service, "YOLO", lambda _path: model)
    with pytest.raises(InvalidYoloModelError, match=message):
        yolo_service.validate_candidate_model(tmp_path / "candidate.pt")
