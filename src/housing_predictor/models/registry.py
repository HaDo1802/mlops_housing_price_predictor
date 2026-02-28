"""MLflow registry helpers."""

import logging
import subprocess
import time
from datetime import datetime, timezone

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ModelRegistryManager:
    """Handles model registration and tagging."""

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

    def register_model(
        self,
        run_id: str,
        model_type: str | None = None,
        model_key: str | None = None,
    ) -> str:
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(
            model_uri=model_uri, name=self.registry_model_name
        )
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
            "model_type": model_type or "unknown",
            "model_key": model_key or "unknown",
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
