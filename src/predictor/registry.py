"""Registry and production artifact helpers."""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "production"


class ModelRegistryManager:
    """Handle model registration, comparison, and promotion."""

    def __init__(self, registry_model_name: str):
        self.registry_model_name = registry_model_name

    def register_model(self, run_id: str) -> str:
        """Register a model artifact from an MLflow run and return its version."""
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri=model_uri, name=self.registry_model_name)
        version = str(result.version)

        client = MlflowClient()
        status = client.get_model_version(self.registry_model_name, version).status
        attempts = 0
        while status == "PENDING_REGISTRATION" and attempts < 30:
            time.sleep(1)
            attempts += 1
            status = client.get_model_version(self.registry_model_name, version).status

        logger.info(
            "Model registered successfully: %s v%s (status=%s)",
            self.registry_model_name,
            version,
            status,
        )
        return version

    def _get_production_metric(
        self,
        client: MlflowClient,
        metric_name: str = "test_r2",
    ) -> float | None:
        try:
            versions = client.search_model_versions(
                f"name='{self.registry_model_name}'"
            )
        except MlflowException as exc:
            if "not found" in str(exc).lower():
                return None
            raise

        production_versions = [
            version
            for version in versions
            if getattr(version, "current_stage", None) == "Production"
        ]
        if not production_versions:
            return None

        production_version = max(production_versions, key=lambda version: int(version.version))

        prod_run_id = production_version.run_id
        prod_run = client.get_run(prod_run_id)
        metric = prod_run.data.metrics.get(metric_name)
        return float(metric) if metric is not None else None

    def evaluate_and_promote(
        self,
        run_id: str,
        current_metric: float,
        metric_name: str = "test_r2",
        improvement_threshold: float = 0.02,
        stage: str = "Production",
    ) -> bool:
        """Register and promote the candidate run if it beats production."""
        client = MlflowClient()
        production_metric = self._get_production_metric(client, metric_name)
        is_better = (
            production_metric is None
            or float(current_metric) - float(production_metric) > float(improvement_threshold)
        )

        logger.info(
            "Promotion gate result: current=%s production=%s passed=%s",
            current_metric,
            production_metric,
            is_better,
        )

        if not is_better:
            return False

        version = self.register_model(run_id)
        archive_existing = stage == "Production"
        client.transition_model_version_stage(
            name=self.registry_model_name,
            version=str(version),
            stage=stage,
            archive_existing_versions=archive_existing,
        )
        return True


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
    """Save local production artifacts directly from in-memory objects."""
    target_dir = _target_dir(output_dir)

    with open(target_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    if hasattr(preprocessor, "save"):
        preprocessor.save(target_dir / "preprocessor.pkl")
    else:
        with open(target_dir / "preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)

    with open(target_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    _write_yaml(target_dir / "config.yaml", config_dict, sort_keys=False)
    return {"output_dir": str(target_dir), "synced": True, "source": "in_memory"}
