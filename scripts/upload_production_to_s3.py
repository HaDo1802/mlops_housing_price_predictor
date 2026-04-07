"""Upload the current local production artifact snapshot to S3."""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from promote import sync_artifacts_to_s3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "production"

logger = logging.getLogger(__name__)

load_dotenv(PROJECT_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the local production model snapshot to S3."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Directory containing model.pkl, preprocessor.pkl, metadata.json, and config.yaml.",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="S3 bucket name. Defaults to ARTIFACT_BUCKET.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    args = parse_args()

    bucket = args.bucket or os.getenv("ARTIFACT_BUCKET")
    if not bucket:
        raise ValueError("Provide --bucket or set ARTIFACT_BUCKET.")

    model_dir = Path(args.model_dir).resolve()
    sync_artifacts_to_s3(model_dir=model_dir, bucket=bucket)
    logger.info("Uploaded production snapshot from %s to bucket %s", model_dir, bucket)


if __name__ == "__main__":
    main()
