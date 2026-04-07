"""Script for MLflow model stage transitions and local production sync."""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
import pickle
import shutil
from pathlib import Path

import boto3
import mlflow.sklearn
import yaml
from dotenv import load_dotenv
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "production"
logger = logging.getLogger(__name__)

load_dotenv(PROJECT_ROOT / ".env")


def list_models(client: MlflowClient, model_name: str | None = None) -> None:
    if model_name:
        registered_models = [client.get_registered_model(model_name)]
    else:
        registered_models = client.search_registered_models()

    print(f"DEBUG: Found {len(registered_models)} registered models")

    if not registered_models:
        print("No registered models found in MLflow registry.")
        return

    for rm in registered_models:
        print(f"\nModel: {rm.name}")
        versions = sorted(rm.latest_versions, key=lambda v: int(v.version))
        print(f"  Versions: {len(versions)}")
        for mv in versions:
            run = client.get_run(mv.run_id)
            print(
                f"  - v{mv.version} | stage={mv.current_stage or 'None'} | "
                f"test_r2={run.data.metrics.get('test_r2')} | run_id={mv.run_id}"
            )


def transition_stage(
    client: MlflowClient, model_name: str, version: str, stage: str
) -> None:
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )


def resolve_version(client: MlflowClient, model_name: str, version: str | None) -> str:
    """Return requested version, or latest registered version when omitted."""
    if version:
        return str(version)

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No versions found for model '{model_name}'.")
    latest = max(versions, key=lambda mv: int(mv.version))
    print(f"No --version supplied. Using latest version: {latest.version}")
    return str(latest.version)


def _download_optional_run_artifact(run_id: str, artifact_path: str) -> str | None:
    """Download artifact from run if it exists; returns local path or None."""
    try:
        return download_artifacts(artifact_uri=f"runs:/{run_id}/{artifact_path}")
    except Exception:
        return None


def sync_local_production_artifacts(
    client: MlflowClient,
    model_name: str,
    version: str,
    output_dir: Path,
) -> None:
    """Materialize promoted model artifacts into local models/production."""
    model_version = client.get_model_version(model_name, version)
    run_id = model_version.run_id

    output_dir.mkdir(parents=True, exist_ok=True)

    model = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    preprocessor_local = _download_optional_run_artifact(run_id, "preprocessor.pkl")
    if preprocessor_local:
        shutil.copy2(preprocessor_local, output_dir / "preprocessor.pkl")
    else:
        raise FileNotFoundError(
            f"Missing preprocessor artifact for run {run_id}: preprocessor.pkl"
        )

    metadata_local = _download_optional_run_artifact(run_id, "metadata.json")
    if metadata_local:
        shutil.copy2(metadata_local, output_dir / "metadata.json")
    else:
        with open(output_dir / "metadata.json", "w") as f:
            json.dump({}, f)

    config_local = _download_optional_run_artifact(run_id, "config/config.yaml")
    if not config_local:
        config_local = _download_optional_run_artifact(run_id, "config.yaml")
    if config_local:
        shutil.copy2(config_local, output_dir / "config.yaml")
    else:
        with open(output_dir / "config.yaml", "w") as f:
            yaml.safe_dump({"note": "No config artifact found in MLflow run."}, f)

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    metadata.update(
        {
            "registry_model_name": model_name,
            "registry_version": str(version),
            "run_id": run_id,
            "promotion_stage": "Production",
            "promotion_timestamp": datetime.now(timezone.utc).isoformat(),
            "artifact_delivery": "local_and_optional_s3_snapshot",
        }
    )
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Synced promoted artifacts to: {output_dir}")


def sync_artifacts_to_s3(model_dir: Path, bucket: str) -> None:
    """Upload the production artifact set to the configured S3 bucket."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote MLflow model versions and sync local production artifacts."
    )
    parser.add_argument(
        "--model-name",
        default="housing_price_predictor",
        help="Registered MLflow model name.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Model version to transition. Defaults to latest registered version.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="Production",
        choices=["Staging", "Production", "Archived"],
        help="Target stage for transition.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list models and versions, without stage transition.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Local directory for synced production artifacts.",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="Optional S3 bucket override for production artifact sync.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = MlflowClient()

    print(f"MLflow Tracking URI: {client.tracking_uri}")
    print("Searching for all registered models...\n")

    list_models(client, model_name=None)

    if args.list_only:
        return

    target_version = resolve_version(client, args.model_name, args.version)

    transition_stage(client, args.model_name, target_version, args.stage)
    list_models(client, model_name=args.model_name)

    if args.stage == "Production":
        sync_local_production_artifacts(
            client=client,
            model_name=args.model_name,
            version=target_version,
            output_dir=Path(args.output_dir),
        )
        bucket = args.bucket or os.getenv("ARTIFACT_BUCKET")
        if bucket:
            sync_artifacts_to_s3(Path(args.output_dir), bucket)
        else:
            logger.warning("ARTIFACT_BUCKET not set; skipping S3 artifact sync.")


if __name__ == "__main__":
    main()
