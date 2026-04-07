"""Production artifact sync helpers."""

import json
import logging
import os
import pickle
import shutil
import time
from pathlib import Path

import boto3
import mlflow
import yaml
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient

from predictor.registry import resolve_version

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "production"


def _target_dir(output_dir: Path | None) -> Path:
    target = output_dir or DEFAULT_OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_yaml(path: Path, payload: dict, sort_keys: bool = False) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=sort_keys)


def _download_optional_run_artifact(run_id: str, artifact_path: str) -> str | None:
    """Download an artifact from an MLflow run if it exists."""
    try:
        return download_artifacts(artifact_uri=f"runs:/{run_id}/{artifact_path}")
    except Exception:
        return None


def sync_production_to_local(
    model_name: str,
    output_dir: Path | None = None,
) -> dict:
    """Materialize the current Production model version into local production files."""
    target_dir = _target_dir(output_dir)
    version = resolve_version(model_name, stage="Production")
    client = MlflowClient()
    model_version = client.get_model_version(model_name, str(version))
    run_id = model_version.run_id

    model = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
    with open(target_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    preprocessor_local = _download_optional_run_artifact(run_id, "preprocessor.pkl")
    if not preprocessor_local:
        raise FileNotFoundError(
            f"Missing preprocessor artifact for run {run_id}: preprocessor.pkl"
        )
    shutil.copy2(preprocessor_local, target_dir / "preprocessor.pkl")

    metadata_local = _download_optional_run_artifact(run_id, "metadata.json")
    if metadata_local:
        shutil.copy2(metadata_local, target_dir / "metadata.json")
    else:
        with open(target_dir / "metadata.json", "w") as f:
            json.dump({}, f)

    config_local = _download_optional_run_artifact(run_id, "config/config.yaml")
    if not config_local:
        config_local = _download_optional_run_artifact(run_id, "config.yaml")
    if config_local:
        shutil.copy2(config_local, target_dir / "config.yaml")
    else:
        _write_yaml(
            target_dir / "config.yaml",
            {"note": "No config artifact found in MLflow run."},
        )

    metadata_path = target_dir / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    metadata.update(
        {
            "registry_model_name": model_name,
            "registry_version": str(version),
            "run_id": run_id,
            "promotion_stage": model_version.current_stage,
            "promotion_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "artifact_delivery": "local_production_snapshot",
        }
    )
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Synced production artifacts locally to %s", target_dir)
    return {
        "output_dir": str(target_dir),
        "run_id": run_id,
        "model_name": model_name,
        "version": str(version),
        "source": "mlflow_registry",
    }


def sync_artifacts_to_s3(model_dir: Path, bucket: str) -> str:
    """Upload the local production artifact set to S3."""
    s3 = boto3.client(
        "s3",
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )
    prefix = "models/production"
    artifact_names = ["model.pkl", "preprocessor.pkl", "metadata.json", "config.yaml"]

    for artifact_name in artifact_names:
        local_path = model_dir / artifact_name
        if not local_path.exists():
            raise FileNotFoundError(f"Missing artifact for S3 sync: {local_path}")
        s3.upload_file(str(local_path), bucket, f"{prefix}/{artifact_name}")

    logger.info("Synced production artifacts to s3://%s/%s", bucket, prefix)
    return f"s3://{bucket}/{prefix}"


def sync_production_to_local_and_s3(
    model_name: str,
    output_dir: Path | None = None,
    bucket: str | None = None,
) -> dict:
    """Sync the current Production model version to local files and optionally S3."""
    result = sync_production_to_local(model_name=model_name, output_dir=output_dir)
    if bucket:
        result["s3_uri"] = sync_artifacts_to_s3(Path(result["output_dir"]), bucket)
    return result
