"""Sync the current Production MLflow model to local files and optionally S3."""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from predictor.artifact_store import sync_production_to_local_and_s3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "production"

load_dotenv(PROJECT_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync the current Production model from MLflow registry to local files and optional S3."
    )
    parser.add_argument(
        "--model-name",
        default="housing_price_predictor",
        help="Registered MLflow model name.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Local destination for model.pkl, preprocessor.pkl, metadata.json, and config.yaml.",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="Optional S3 bucket override. Defaults to ARTIFACT_BUCKET.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    bucket = args.bucket or os.getenv("ARTIFACT_BUCKET")
    result = sync_production_to_local_and_s3(
        model_name=args.model_name,
        output_dir=Path(args.output_dir),
        bucket=bucket,
    )
    print(result)


if __name__ == "__main__":
    main()
