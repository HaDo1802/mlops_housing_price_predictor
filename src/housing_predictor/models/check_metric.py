"""Metric gate helpers for deciding whether a run should be promoted."""

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

from mlflow.tracking import MlflowClient


@dataclass
class MetricCheckResult:
    should_register: bool
    metric_name: str
    candidate_metric: float
    production_metric: float | None
    improvement: float | None
    threshold: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_production_metric(
    client: MlflowClient,
    model_name: str,
    metric_name: str = "test_r2",
) -> float | None:
    """Fetch current production metric from MLflow registry model."""
    production_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not production_versions:
        return None

    prod_run_id = production_versions[0].run_id
    prod_run = client.get_run(prod_run_id)
    metric = prod_run.data.metrics.get(metric_name)
    return float(metric) if metric is not None else None


def check_metric_against_production(
    client: MlflowClient,
    model_name: str,
    candidate_metric: float,
    metric_name: str = "test_r2",
    improvement_threshold: float = 0.02,
) -> MetricCheckResult:
    """
    Decide whether candidate should be registered/promoted.

    Rule:
    - If no production model exists, candidate passes.
    - Otherwise, candidate must exceed production metric by threshold.
    """
    production_metric = get_production_metric(client, model_name, metric_name)

    if production_metric is None:
        return MetricCheckResult(
            should_register=True,
            metric_name=metric_name,
            candidate_metric=float(candidate_metric),
            production_metric=None,
            improvement=None,
            threshold=float(improvement_threshold),
            reason="No production model/metric found; candidate accepted.",
        )

    improvement = float(candidate_metric) - float(production_metric)
    should_register = improvement > float(improvement_threshold)
    reason = (
        "Candidate beats production threshold."
        if should_register
        else "Candidate does not beat production threshold."
    )

    return MetricCheckResult(
        should_register=should_register,
        metric_name=metric_name,
        candidate_metric=float(candidate_metric),
        production_metric=float(production_metric),
        improvement=float(improvement),
        threshold=float(improvement_threshold),
        reason=reason,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether a candidate run beats current Production metric."
    )
    parser.add_argument("--run-id", required=True, help="Candidate MLflow run id.")
    parser.add_argument(
        "--model-name",
        default="housing_price_predictor",
        help="MLflow registered model name.",
    )
    parser.add_argument(
        "--metric-name",
        default="test_r2",
        help="Metric key used for comparison.",
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.02,
        help="Required improvement over current Production metric.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = MlflowClient()
    run = client.get_run(args.run_id)
    if args.metric_name not in run.data.metrics:
        raise ValueError(f"Metric '{args.metric_name}' missing in run {args.run_id}")

    result = check_metric_against_production(
        client=client,
        model_name=args.model_name,
        candidate_metric=float(run.data.metrics[args.metric_name]),
        metric_name=args.metric_name,
        improvement_threshold=args.improvement_threshold,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
