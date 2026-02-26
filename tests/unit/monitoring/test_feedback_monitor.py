import pandas as pd

from pipelines.run_feedback_monitor import compute_psi, evaluate_feedback


def test_compute_psi_zero_for_identical_distributions():
    reference = pd.Series([1, 2, 3, 4, 5] * 10)
    current = pd.Series([1, 2, 3, 4, 5] * 10)

    psi = compute_psi(reference, current)
    assert psi is not None
    assert abs(psi) < 1e-6


def test_compute_psi_returns_none_for_constant_series():
    reference = pd.Series([5] * 20)
    current = pd.Series([5] * 20)

    psi = compute_psi(reference, current)
    assert psi is None


def test_evaluate_feedback_metrics():
    df = pd.DataFrame(
        {
            "agree_with_prediction": [True, False, True, False],
            "predicted_price": [200000, 250000, 210000, 260000],
            "suggested_min": [None, 240000, None, 255000],
            "suggested_max": [None, 260000, None, 275000],
        }
    )

    metrics = evaluate_feedback(df)

    assert metrics["total_feedback"] == 4
    assert metrics["agree_rate"] == 0.5
    assert "pred_within_user_range_rate" in metrics
    assert "avg_user_range_width" in metrics
