"""Register and promote MLflow candidates using registry.py helpers."""

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from predictor.registry import (
    evaluate_and_promote,
    list_versions,
    promote_version,
    register_model,
    resolve_version,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)

load_dotenv(PROJECT_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register and promote MLflow candidate runs or model versions."
    )
    parser.add_argument(
        "--model-name",
        default="housing_price_predictor",
        help="Registered MLflow model name.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="MLflow run ID to evaluate, register, and promote.",
    )
    parser.add_argument(
        "--metric",
        type=float,
        default=None,
        help="Candidate metric for the run ID, usually test_r2.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Existing registered model version to promote directly.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="Production",
        choices=["Staging", "Production", "Archived"],
        help="Target stage for transition.",
    )
    parser.add_argument("--list-only", action="store_true", help="List versions only.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    args = parse_args()

    if args.list_only:
        versions = list_versions(args.model_name)
        if not versions:
            print("No registered model versions found.")
            return
        for item in versions:
            print(
                f"v{item.version} | stage={item.current_stage or 'None'} | run_id={item.run_id}"
            )
        return

    if args.run_id:
        if args.metric is None:
            client = MlflowClient()
            run = client.get_run(args.run_id)
            inferred_metric = run.data.metrics.get("test_r2")

            if inferred_metric is not None:
                args.metric = float(inferred_metric)
            else:
                versions = list_versions(args.model_name)
                production_versions = [
                    item
                    for item in versions
                    if getattr(item, "current_stage", None) == "Production"
                ]
                if not production_versions:
                    version = register_model(args.run_id, args.model_name)
                    promote_version(
                        model_name=args.model_name,
                        version=version,
                        stage=args.stage,
                    )
                    print(
                        {
                            "passed": True,
                            "registered": True,
                            "promoted": True,
                            "version": str(version),
                            "production_metric": None,
                            "candidate_metric": None,
                        }
                    )
                    return
                raise ValueError(
                    "--metric was not provided and the run does not contain 'test_r2'. "
                    "Pass --metric explicitly for comparison against the current production model."
                )
        result = evaluate_and_promote(
            model_name=args.model_name,
            run_id=args.run_id,
            current_metric=args.metric,
            stage=args.stage,
        )
        print(result)
        return

    target_version = resolve_version(args.model_name, version=args.version)
    promote_version(model_name=args.model_name, version=target_version, stage=args.stage)
    print(
        {
            "model_name": args.model_name,
            "version": target_version,
            "stage": args.stage,
            "promoted": True,
        }
    )


if __name__ == "__main__":
    main()
