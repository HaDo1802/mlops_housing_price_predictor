"""Triggered promotion DAG for model governance."""

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
    logging.info("Trigger run_id: %s", conf.get("run_id"))
    logging.info("Trigger test_r2: %s", conf.get("test_r2"))
    logging.info("Triggered by: %s", conf.get("triggered_by"))


def evaluate_and_promote_task(**context) -> None:
    from predictor.registry import evaluate_and_promote

    conf = (context.get("dag_run").conf if context.get("dag_run") else None) or {}
    run_id = conf.get("run_id")
    test_r2 = conf.get("test_r2")
    if not run_id:
        raise ValueError("promote_candidate_dag requires dag_run.conf['run_id'].")
    if test_r2 is None:
        raise ValueError("promote_candidate_dag requires dag_run.conf['test_r2'].")

    result = evaluate_and_promote(
        model_name="housing_price_predictor",
        run_id=run_id,
        current_metric=float(test_r2),
        metric_name="test_r2",
        improvement_threshold=0.02,
        stage="Production",
    )
    context["ti"].xcom_push(key="promotion_result", value=result)

    if result.get("promoted", False):
        logging.info("Candidate promoted successfully: %s", result)
    else:
        logging.info("Candidate was not promoted: %s", result)


def decide_sync_task(**context) -> bool:
    result = context["ti"].xcom_pull(
        task_ids="evaluate_and_promote",
        key="promotion_result",
    ) or {}
    promoted = bool(result.get("promoted", False))
    if not promoted:
        logging.info(
            "Skipping sync because candidate did not beat production. result=%s",
            result,
        )
    return promoted


def sync_artifacts_task(**context) -> None:
    from predictor.artifact_store import sync_production_to_local_and_s3

    bucket = os.getenv("ARTIFACT_BUCKET")
    if not bucket:
        raise RuntimeError(
            "promote_candidate_dag requires ARTIFACT_BUCKET so promoted artifacts can be synced to S3."
        )

    result = sync_production_to_local_and_s3(
        model_name="housing_price_predictor",
        output_dir=PROJECT_ROOT / "models" / "production",
        bucket=bucket,
    )
    context["ti"].xcom_push(key="sync_result", value=result)
    logging.info("Production sync complete. output_dir=%s", result.get("output_dir"))
    logging.info("Production sync complete. s3_uri=%s", result.get("s3_uri"))


def build_promote_candidate_dag():
    from airflow import DAG
    from airflow.operators.python import PythonOperator, ShortCircuitOperator

    default_args = {
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    with DAG(
        dag_id="promote_candidate_dag",
        default_args=default_args,
        start_date=datetime(2025, 1, 1),
        schedule=None,
        catchup=False,
        tags=["mlops", "promotion"],
        render_template_as_native_obj=True,
    ) as dag:
        log_trigger_context = PythonOperator(
            task_id="log_trigger_context",
            python_callable=log_trigger_context_task,
        )

        evaluate_and_promote_op = PythonOperator(
            task_id="evaluate_and_promote",
            python_callable=evaluate_and_promote_task,
        )

        decide_sync = ShortCircuitOperator(
            task_id="decide_sync",
            python_callable=decide_sync_task,
        )

        sync_artifacts = PythonOperator(
            task_id="sync_artifacts",
            python_callable=sync_artifacts_task,
        )

        log_trigger_context >> evaluate_and_promote_op >> decide_sync >> sync_artifacts

    return dag


dag = build_promote_candidate_dag()
