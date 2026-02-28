"""Promotion helpers for MLflow registered model versions."""

import json
import pickle
from pathlib import Path
from typing import Any

import yaml
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "production"


def _target_dir(output_dir: Path | None) -> Path:
    target = output_dir or DEFAULT_OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_yaml(path: Path, payload: dict[str, Any], sort_keys: bool = False) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=sort_keys)


def save_local_production_from_objects(
    model: Any,
    preprocessor: Any,
    metadata: dict[str, Any],
    config_dict: dict[str, Any],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Save local production artifacts directly from in-memory objects.

    This is intended for training-time promotion where model/preprocessor are
    already loaded in memory and we want to avoid re-downloading from MLflow.
    """
    target_dir = _target_dir(output_dir)

    with open(target_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    if hasattr(preprocessor, "save"):
        preprocessor.save(str(target_dir / "preprocessor.pkl"))
    else:
        with open(target_dir / "preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)

    with open(target_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    _write_yaml(target_dir / "config.yaml", config_dict, sort_keys=False)

    return {"output_dir": str(target_dir), "synced": True, "source": "in_memory"}


def promote_model_version(
    client: MlflowClient,
    model_name: str,
    version: str,
    stage: str = "Production",
) -> dict[str, Any]:
    """Transition a registered model version to a target stage."""
    archive_existing = stage == "Production"
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=stage,
        archive_existing_versions=archive_existing,
    )
    result: dict[str, Any] = {
        "model_name": model_name,
        "version": str(version),
        "stage": stage,
        "archive_existing_versions": archive_existing,
        "promoted": True,
    }
    return result
