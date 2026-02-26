"""MLflow registry and promotion helpers."""

import logging
import subprocess
import time
from datetime import datetime, timezone

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ModelRegistryManager:
    """Handles model registration, tagging, and promotion."""

    def __init__(self, registry_model_name: str):
        self.registry_model_name = registry_model_name

    @staticmethod
    def get_git_commit() -> str:
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode("utf-8")
                .strip()
            )
        except Exception:
            return "unknown"

    def register_model(self, run_id: str) -> str:
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

        version_tags = {
            "training_date": datetime.now(timezone.utc).isoformat(),
            "git_commit": self.get_git_commit(),
            "model_type": "GradientBoostingRegressor",
        }
        for key, value in version_tags.items():
            client.set_model_version_tag(
                name=self.registry_model_name,
                version=version,
                key=key,
                value=str(value),
            )

        logger.info(
            "Model registered successfully: %s v%s (status=%s)",
            self.registry_model_name,
            version,
            status,
        )
        return version

    def auto_promote_if_better(
        self,
        new_model_version: str,
        new_test_r2: float,
        improvement_threshold: float = 0.02,
    ) -> None:
        client = MlflowClient()
        production_versions = client.get_latest_versions(
            self.registry_model_name, stages=["Production"]
        )

        if not production_versions:
            client.transition_model_version_stage(
                name=self.registry_model_name,
                version=new_model_version,
                stage="Staging",
            )
            client.transition_model_version_stage(
                name=self.registry_model_name,
                version=new_model_version,
                stage="Production",
                archive_existing_versions=True,
            )
            return

        current_prod = production_versions[0]
        prod_run = client.get_run(current_prod.run_id)
        prod_r2 = prod_run.data.metrics.get("test_r2")

        if prod_r2 is None:
            client.transition_model_version_stage(
                name=self.registry_model_name,
                version=new_model_version,
                stage="Staging",
            )
            return

        improvement = new_test_r2 - float(prod_r2)
        target_stage = "Production" if improvement > improvement_threshold else "Staging"

        if target_stage == "Production":
            client.transition_model_version_stage(
                name=self.registry_model_name,
                version=new_model_version,
                stage="Staging",
            )
            client.transition_model_version_stage(
                name=self.registry_model_name,
                version=new_model_version,
                stage="Production",
                archive_existing_versions=True,
            )
        else:
            client.transition_model_version_stage(
                name=self.registry_model_name,
                version=new_model_version,
                stage="Staging",
            )
