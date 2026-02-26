import json

import pandas as pd

from src.housing_predictor.monitoring.feedback_collector import (
    load_feedback,
    save_feedback_record,
)


def test_save_and_load_feedback_roundtrip(tmp_path):
    output_path = tmp_path / "feedback.csv"
    record = {
        "feedback_id": "fb-1",
        "prediction_id": "pred-1",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "agree_with_prediction": True,
        "predicted_price": 250000.0,
        "lower_bound": 230000.0,
        "upper_bound": 270000.0,
        "suggested_min": None,
        "suggested_max": None,
        "input_features": json.dumps({"Lot Area": 9000, "Neighborhood": "NAmes"}),
    }

    save_feedback_record(record, output_path=str(output_path))
    df = load_feedback(output_path=str(output_path))

    assert "Lot Area" in df.columns
    assert "Neighborhood" in df.columns
    assert df.loc[0, "Neighborhood"] == "NAmes"
    assert df.loc[0, "predicted_price"] == 250000.0


def test_save_feedback_appends_records(tmp_path):
    output_path = tmp_path / "feedback.csv"
    record_a = {
        "feedback_id": "fb-1",
        "prediction_id": "pred-1",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "agree_with_prediction": True,
        "predicted_price": 250000.0,
        "lower_bound": 230000.0,
        "upper_bound": 270000.0,
        "suggested_min": None,
        "suggested_max": None,
        "input_features": json.dumps({"Lot Area": 9000}),
    }
    record_b = {
        "feedback_id": "fb-2",
        "prediction_id": "pred-2",
        "timestamp_utc": "2026-01-02T00:00:00+00:00",
        "agree_with_prediction": False,
        "predicted_price": 300000.0,
        "lower_bound": 280000.0,
        "upper_bound": 320000.0,
        "suggested_min": 290000.0,
        "suggested_max": 310000.0,
        "input_features": json.dumps({"Lot Area": 9500}),
    }

    save_feedback_record(record_a, output_path=str(output_path))
    save_feedback_record(record_b, output_path=str(output_path))

    df = pd.read_csv(output_path)
    assert len(df) == 2
