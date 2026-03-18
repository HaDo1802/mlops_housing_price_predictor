"""Triggered retraining DAG for housing price model.

This DAG is not scheduled and runs only when triggered by
`data_ingestion_dag`. Training is executed as step-wise tasks (load, split,
preprocess, train, save) to improve observability and debugging in Airflow.

To re-enable real drift gating in the upstream DAG, uncomment the disabled
PSI block in `dags/data_ingestion_dag.py`.
"""

# Required env vars: SUPABASE_DB_HOST, SUPABASE_DB_USER,
#                    SUPABASE_DB_PASSWORD, SUPABASE_DB_PORT (optional)

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path("/opt/airflow/project")
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)


def log_trigger_context_task(**context) -> None:
    conf = (context.get("dag_run").conf if context.get("dag_run") else None) or {}

    if not conf:
        logging.warning("No dag_run.conf provided; proceeding with default context.")
        return

    logging.info("Trigger dataset_path: %s", conf.get("dataset_path"))
    logging.info("Triggered by: %s", conf.get("triggered_by"))
    logging.info("Drift summary: %s", conf.get("drift_summary"))


def _build_pipeline():
    from pathlib import Path

    from housing_predictor.pipelines.training import TrainingPipeline

    # Ensure relative paths in config resolve from project root in container.
    os.chdir(PROJECT_ROOT)

    config_path = str(PROJECT_ROOT / "conf" / "config.yaml")
    pipeline = TrainingPipeline(config_path)

    # Normalize configured data path to an absolute path for Airflow containers.
    raw_path = Path(pipeline.config.data.raw_data_path)
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Training raw_data_path not found: {raw_path} "
            f"(from config: {pipeline.config.data.raw_data_path})"
        )
    pipeline.config.data.raw_data_path = str(raw_path)
    return pipeline


def validate_training_inputs_task(**context) -> None:
    pipeline = _build_pipeline()
    logging.info("Config path: %s", str(PROJECT_ROOT / "conf" / "config.yaml"))
    logging.info("Using training dataset: %s", pipeline.config.data.raw_data_path)
    context["ti"].xcom_push(
        key="training_dataset_path", value=pipeline.config.data.raw_data_path
    )


def load_and_select_task(**context) -> None:
    pipeline = _build_pipeline()
    pipeline.load_and_select()
    logging.info(
        "Loaded rows=%s cols=%s",
        len(pipeline.df_selected),
        len(pipeline.df_selected.columns),
    )


def split_data_task(**context) -> None:
    pipeline = _build_pipeline()
    pipeline.load_and_select()
    pipeline.split_data()
    logging.info(
        "Train rows=%s | Test rows=%s", len(pipeline.X_train), len(pipeline.X_test)
    )


def preprocess_data_task(**context) -> None:
    pipeline = _build_pipeline()
    pipeline.load_and_select()
    pipeline.split_data()
    pipeline.preprocess_data()
    logging.info(
        "Preprocessed shapes | train=%s test=%s",
        getattr(pipeline.X_train_transformed, "shape", None),
        getattr(pipeline.X_test_transformed, "shape", None),
    )


def train_and_save_task(**context) -> None:
    pipeline = _build_pipeline()
    pipeline.load_and_select()
    pipeline.split_data()
    pipeline.preprocess_data()
    pipeline.train_and_eval()

    artifact_dir = PROJECT_ROOT / "models" / "experiments" / "airflow_latest"
    saved_path = pipeline.save_artifacts(output_dir=str(artifact_dir))

    metrics = pipeline.metrics or {}
    metrics["promote_to_production"] = False
    metrics["artifact_path"] = str(saved_path)

    test_metrics = metrics.get("test", {}) if isinstance(metrics, dict) else {}
    logging.info("test_r2=%s", test_metrics.get("r2"))
    logging.info("test_rmse=%s", test_metrics.get("rmse"))
    logging.info("test_mae=%s", test_metrics.get("mae"))
    logging.info("promote_to_production=%s", metrics.get("promote_to_production"))

    context["ti"].xcom_push(key="training_metrics", value=metrics)


def notify_result_task(**context) -> None:
    metrics = (
        context["ti"].xcom_pull(task_ids="train_and_save", key="training_metrics") or {}
    )

    conf = (context.get("dag_run").conf if context.get("dag_run") else None) or {}
    drift_summary = conf.get("drift_summary") or {}

    test_metrics = metrics.get("test", {}) if isinstance(metrics, dict) else {}

    logging.info("=== RETRAINING COMPLETE ===")
    logging.info("Triggered by      : %s", conf.get("triggered_by", "unknown"))
    logging.info("Drift level       : %s", drift_summary.get("drift_level"))
    logging.info("Drift PSI         : %s", drift_summary.get("psi"))
    logging.info("Drift reason      : %s", drift_summary.get("reason"))
    logging.info("Test R²           : %s", test_metrics.get("r2"))
    logging.info("Test RMSE         : %s", test_metrics.get("rmse"))
    logging.info("Test MAE          : %s", test_metrics.get("mae"))
    logging.info("Promoted to prod  : %s", metrics.get("promote_to_production"))
    logging.info("===========================")


def build_retraining_dag():
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    default_args = {
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    }

    with DAG(
        dag_id="retraining_dag",
        default_args=default_args,
        start_date=datetime(2025, 1, 1),
        schedule=None,
        catchup=False,
        tags=["mlops", "housing_predictor", "training"],
    ) as dag:
        log_trigger_context = PythonOperator(
            task_id="log_trigger_context",
            python_callable=log_trigger_context_task,
        )

        validate_training_inputs = PythonOperator(
            task_id="validate_training_inputs",
            python_callable=validate_training_inputs_task,
        )

        load_and_select = PythonOperator(
            task_id="load_and_select",
            python_callable=load_and_select_task,
        )

        split_data = PythonOperator(
            task_id="split_data",
            python_callable=split_data_task,
        )

        preprocess_data = PythonOperator(
            task_id="preprocess_data",
            python_callable=preprocess_data_task,
        )

        train_and_save = PythonOperator(
            task_id="train_and_save",
            python_callable=train_and_save_task,
        )

        notify_result = PythonOperator(
            task_id="notify_result",
            python_callable=notify_result_task,
        )

        (
            log_trigger_context
            >> validate_training_inputs
            >> load_and_select
            >> split_data
            >> preprocess_data
            >> train_and_save
            >> notify_result
        )

    return dag


dag = build_retraining_dag()
