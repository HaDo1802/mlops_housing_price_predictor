"""Triggered training DAG for candidate model creation."""

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
    logging.info("Trigger dataset_path: %s", conf.get("dataset_path"))
    logging.info("Triggered by: %s", conf.get("triggered_by"))
    logging.info("Drift summary: %s", conf.get("drift_summary"))


def run_training_task(**context) -> None:
    import mlflow
    from predictor.training_pipeline import TrainingPipeline

    os.chdir(PROJECT_ROOT)

    conf = (context.get("dag_run").conf if context.get("dag_run") else None) or {}
    dataset_path = conf.get("dataset_path")

    pipeline = TrainingPipeline("conf/config.yaml")
    if dataset_path:
        pipeline.config.data.raw_data_path = str(dataset_path)
        logging.info("Using dataset_path from dag_run.conf: %s", dataset_path)
    pipeline.run(track=True, promote=False)

    run = mlflow.last_active_run()
    if run is None:
        raise RuntimeError("MLflow did not return a last_active_run after training.")

    run_id = run.info.run_id
    test_metrics = pipeline.metrics["test"]
    test_r2 = float(test_metrics["r2"])

    logging.info("Training complete. run_id=%s", run_id)
    logging.info("test_r2=%s", test_r2)
    logging.info("test_rmse=%s", test_metrics["rmse"])
    logging.info("test_mae=%s", test_metrics["mae"])

    context["ti"].xcom_push(key="run_id", value=run_id)
    context["ti"].xcom_push(key="test_r2", value=test_r2)


def build_train_candidate_dag():
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator

    default_args = {
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    with DAG(
        dag_id="train_candidate_dag",
        default_args=default_args,
        start_date=datetime(2025, 1, 1),
        schedule=None,
        catchup=False,
        tags=["mlops", "training"],
        render_template_as_native_obj=True,
    ) as dag:
        log_trigger_context = PythonOperator(
            task_id="log_trigger_context",
            python_callable=log_trigger_context_task,
        )

        run_training = PythonOperator(
            task_id="run_training",
            python_callable=run_training_task,
        )

        trigger_promotion = TriggerDagRunOperator(
            task_id="trigger_promotion",
            trigger_dag_id="promote_candidate_dag",
            wait_for_completion=False,
            conf={
                "run_id": "{{ ti.xcom_pull(task_ids='run_training', key='run_id') }}",
                "test_r2": "{{ ti.xcom_pull(task_ids='run_training', key='test_r2') }}",
                "triggered_by": "train_candidate_dag",
            },
        )

        log_trigger_context >> run_training >> trigger_promotion

    return dag


dag = build_train_candidate_dag()
