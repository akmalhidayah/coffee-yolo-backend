from app.services.yolo_service import aggregate_detections


def detection(class_id, class_name, coffee_type, grade, confidence):
    return {
        "class_id": class_id,
        "class_name": class_name,
        "coffee_type": coffee_type,
        "grade": grade,
        "confidence": confidence,
    }


def test_majority_beats_single_highest_confidence():
    result = aggregate_detections(
        [detection(0, "Arabica Grade A", "Arabica", "Grade A", 0.95)]
        + [detection(2, "Arabica Grade C", "Arabica", "Grade C", 0.80)] * 20
    )
    assert result["class_name"] == "Arabica Grade C"
    assert result["summary"]["dominant_count"] == 20
    assert result["summary"]["low_quality_percentage"] == 95.2


def test_tie_uses_total_confidence_then_lowest_class_id():
    confidence_winner = aggregate_detections([
        detection(0, "Arabica Grade A", "Arabica", "Grade A", 0.70),
        detection(1, "Arabica Grade B", "Arabica", "Grade B", 0.80),
    ])
    assert confidence_winner["class_name"] == "Arabica Grade B"

    id_winner = aggregate_detections([
        detection(0, "Arabica Grade A", "Arabica", "Grade A", 0.80),
        detection(1, "Arabica Grade B", "Arabica", "Grade B", 0.80),
    ])
    assert id_winner["class_name"] == "Arabica Grade A"


def test_percentages_and_empty_summary():
    result = aggregate_detections([
        detection(0, "Arabica Grade A", "Arabica", "Grade A", 0.9),
        detection(1, "Arabica Grade B", "Arabica", "Grade B", 0.8),
        detection(2, "Arabica Grade C", "Arabica", "Grade C", 0.7),
        detection(2, "Arabica Grade C", "Arabica", "Grade C", 0.6),
    ])
    assert result["summary"]["low_quality_percentage"] == 50.0
    assert result["summary"]["sorting_required_percentage"] == 75.0
    empty = aggregate_detections([])["summary"]
    assert empty["total"] == 0
    assert len(empty["class_counts"]) == 6
    assert all(value == 0 for value in empty["class_counts"].values())
