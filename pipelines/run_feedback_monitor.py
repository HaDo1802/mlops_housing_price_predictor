"""
Feedback monitoring and simple drift analysis.

Usage:
    python pipelines/run_feedback_monitor.py
"""

import json
from pathlib import Path

import pandas as pd

from conf.config_manager import ConfigManager
from src.housing_predictor.monitoring.drift import compute_psi, evaluate_feedback
from src.housing_predictor.monitoring.feedback_collector import load_feedback


def main() -> None:
    config = ConfigManager("conf/config.yaml").get_config()
    feedback_df = load_feedback()

    if feedback_df.empty:
        print("No feedback data found.")
        return

    # Load reference dataset for drift
    raw_data_path = Path(config.data.raw_data_path)
    if not raw_data_path.exists():
        print(f"Reference data not found: {raw_data_path}")
        return

    reference_df = pd.read_csv(raw_data_path)

    # Compute feedback metrics
    metrics = evaluate_feedback(feedback_df)

    # Compute drift for numeric features
    drift = {}
    numeric_features = config.features.numeric
    for feature in numeric_features:
        if feature in reference_df.columns and feature in feedback_df.columns:
            psi = compute_psi(reference_df[feature], feedback_df[feature])
            if psi is not None:
                drift[feature] = psi

    report = {
        "feedback_metrics": metrics,
        "numeric_feature_psi": drift,
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
