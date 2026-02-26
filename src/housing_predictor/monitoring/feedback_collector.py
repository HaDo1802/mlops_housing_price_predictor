"""Feedback collection utilities."""

import json
from pathlib import Path
from typing import Dict

import pandas as pd


def save_feedback_record(record: Dict, output_path: str = "data/feedback/feedback.csv") -> None:
    """Append a feedback record to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([record])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def load_feedback(output_path: str = "data/feedback/feedback.csv") -> pd.DataFrame:
    """Load feedback records from CSV."""
    path = Path(output_path)
    if not path.exists():
        raise FileNotFoundError(f"Feedback file not found: {path}")

    df = pd.read_csv(path)
    if "input_features" in df.columns:
        features_df = df["input_features"].apply(json.loads).apply(pd.Series)
        df = pd.concat([df.drop(columns=["input_features"]), features_df], axis=1)
    return df
