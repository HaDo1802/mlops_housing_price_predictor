"""Script for MLflow model stage transitions."""

from mlflow.tracking import MlflowClient

# Set these values directly before running this script.
MODEL_NAME = "housing_price_predictor"
VERSION= 11 #str | None = None
STAGE = "Production"#str | None = None  # One of: "Staging", "Production", "Archived"
LIST_ONLY = False


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


def main() -> None:
    client = MlflowClient()

    # Debug: Check MLflow connection
    print(f"MLflow Tracking URI: {client.tracking_uri}")
    print(f"Searching for all registered models...\n")

    list_models(client, model_name=None)

    if LIST_ONLY:
        return

    if VERSION and STAGE:
        transition_stage(client, MODEL_NAME, str(VERSION), STAGE)
        list_models(client, model_name=MODEL_NAME)


if __name__ == "__main__":
    main()
