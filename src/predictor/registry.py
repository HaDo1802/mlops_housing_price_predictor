"""Registry governance helpers."""

import logging
import time
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def list_versions(model_name: str) -> list:
    """Return all registered versions for a model."""
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except MlflowException as exc:
        if "not found" in str(exc).lower():
            return []
        raise
    return sorted(versions, key=lambda version: int(version.version))


def register_model(run_id: str, model_name: str) -> str:
    """Register a model artifact from an MLflow run and return its version."""
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = str(result.version)

    client = MlflowClient()
    status = client.get_model_version(model_name, version).status
    attempts = 0
    while status == "PENDING_REGISTRATION" and attempts < 30:
        time.sleep(1)
        attempts += 1
        status = client.get_model_version(model_name, version).status

    logger.info(
        "Model registered successfully: %s v%s (status=%s)",
        model_name,
        version,
        status,
    )
    return version


def _get_production_metric(
    model_name: str,
    client: MlflowClient,
    metric_name: str = "test_r2",
) -> float | None:
    versions = list_versions(model_name)
    production_versions = [
        version
        for version in versions
        if getattr(version, "current_stage", None) == "Production"
    ]
    if not production_versions:
        return None

    production_version = max(production_versions, key=lambda version: int(version.version))
    prod_run = client.get_run(production_version.run_id)
    metric = prod_run.data.metrics.get(metric_name)
    return float(metric) if metric is not None else None


def promote_version(
    model_name: str,
    version: str,
    stage: str = "Production",
    client: MlflowClient | None = None,
) -> str:
    """Promote an existing registered model version to the requested stage."""
    client = client or MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )
    logger.info("Promoted model %s version %s to stage %s", model_name, version, stage)
    return str(version)


def resolve_version(model_name: str, version: str | None = None, stage: str | None = None) -> str:
    """Return the requested version or resolve one from the registry."""
    if version:
        return str(version)

    versions = list_versions(model_name)
    if stage:
        versions = [item for item in versions if getattr(item, "current_stage", None) == stage]
    if not versions:
        raise ValueError(
            f"No versions found for model '{model_name}'"
            + (f" at stage '{stage}'." if stage else ".")
        )
    latest = max(versions, key=lambda item: int(item.version))
    return str(latest.version)


def evaluate_and_promote(
    model_name: str,
    run_id: str,
    current_metric: float,
    metric_name: str = "test_r2",
    improvement_threshold: float = 0.02,
    stage: str = "Production",
) -> dict[str, Any]:
    """Register and promote the candidate run if it beats production."""
    client = MlflowClient()
    production_metric = _get_production_metric(model_name, client, metric_name)
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
        return {
            "passed": False,
            "registered": False,
            "promoted": False,
            "version": None,
            "production_metric": production_metric,
        }

    version = register_model(run_id, model_name)
    promote_version(model_name=model_name, version=version, stage=stage, client=client)
    return {
        "passed": True,
        "registered": True,
        "promoted": True,
        "version": str(version),
        "production_metric": production_metric,
    }
