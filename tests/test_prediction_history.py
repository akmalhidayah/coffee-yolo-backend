import json

from app.services import user_service


def test_history_stores_top_level_aggregate(tmp_path, monkeypatch):
    monkeypatch.setattr(user_service.settings, "data_dir", tmp_path)
    monkeypatch.setattr(user_service.settings, "database_path", tmp_path / "users.db")
    user_service.record_prediction_history(
        user_id=None,
        image_filename="image.jpg",
        response_status="detected",
        prediction={
            "class_name": "Arabica Grade C",
            "coffee_type": "Arabica",
            "grade": "Grade C",
            "confidence": 0.8,
            "bounding_boxes": [],
            "detections": [{"class_name": "Arabica Grade A", "confidence": 0.99}],
        },
    )
    with user_service._connect() as connection:
        row = connection.execute("SELECT * FROM predictions").fetchone()
    assert row["class_name"] == "Arabica Grade C"
    assert row["confidence"] == 0.8
    assert json.loads(row["detections"])[0]["confidence"] == 0.99
