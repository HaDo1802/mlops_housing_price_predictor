"""CLI for MLflow model stage transitions."""

import argparse

from mlflow.tracking import MlflowClient


def list_models(client: MlflowClient, model_name: str | None = None) -> None:
    if model_name:
        registered_models = [client.get_registered_model(model_name)]
    else:
        registered_models = client.search_registered_models()

    for rm in registered_models:
        print(f"\nModel: {rm.name}")
        versions = sorted(rm.latest_versions, key=lambda v: int(v.version))
        for mv in versions:
            run = client.get_run(mv.run_id)
            print(
                f"  - v{mv.version} | stage={mv.current_stage or 'None'} | "
                f"test_r2={run.data.metrics.get('test_r2')} | run_id={mv.run_id}"
            )


def transition_stage(client: MlflowClient, model_name: str, version: str, stage: str) -> None:
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote/demote MLflow model versions")
    parser.add_argument("--model-name", default="housing_price_predictor")
    parser.add_argument("--version")
    parser.add_argument("--stage", choices=["Staging", "Production", "Archived"])
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = MlflowClient()

    list_models(client, model_name=None)

    if args.list_only:
        return

    if args.version and args.stage:
        transition_stage(client, args.model_name, str(args.version), args.stage)
        list_models(client, model_name=args.model_name)


if __name__ == "__main__":
    main()
