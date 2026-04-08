"""Daily ingestion DAG with drift gate and conditional training trigger."""

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


def ingest_data_task(**context) -> None:
    from airflow.exceptions import AirflowSkipException
    from predictor.config import ConfigManager
    from predictor.data_ingest import DataIngestor

    required = [
        "SUPABASE_DB_HOST",
        "SUPABASE_DB_USER",
        "SUPABASE_DB_PASSWORD",
    ]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise AirflowSkipException(
            "Skipping ingestion because required Supabase env vars are missing: "
            + ", ".join(missing)
        )

    output_path = PROJECT_ROOT / "data" / "raw" / "data_master.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = ConfigManager(str(PROJECT_ROOT / "conf" / "config.yaml")).config
    ingestor = DataIngestor(config)
    df = ingestor.fetch_data()
    df.to_csv(output_path, index=False)

    row_count = int(len(df))
    logging.info("Ingestion complete. Saved %s rows to %s", row_count, output_path)
    context["ti"].xcom_push(key="row_count", value=row_count)


def check_drift_task(**context) -> None:
    import pandas as pd
    from predictor.drift import PSI_THRESHOLD_MODERATE, run_drift_check

    dataset_path = PROJECT_ROOT / "data" / "raw" / "data_master.csv"
    reference_path = PROJECT_ROOT / "data" / "snapshots" / "drift_reference.json"

    try:
        current_df = pd.read_csv(dataset_path)
        result = run_drift_check(
            current_df=current_df,
            reference_path=reference_path,
            psi_threshold=PSI_THRESHOLD_MODERATE,
        )
        should_retrain = bool(result.get("should_retrain", False))
        drift_summary = result
        logging.info(
            "Drift check complete. should_retrain=%s summary=%s",
            should_retrain,
            drift_summary,
        )
    except Exception as exc:
        should_retrain = False
        drift_summary = {
            "should_retrain": False,
            "reason": f"Drift check failed: {exc}",
        }
        logging.warning("Drift check failed. Retraining skipped. Error: %s", exc)

    context["ti"].xcom_push(key="should_retrain", value=should_retrain)
    context["ti"].xcom_push(key="drift_summary", value=drift_summary)


def decide_retrain_task(**context) -> bool:
    return bool(
        context["ti"].xcom_pull(task_ids="check_drift", key="should_retrain")
    )


def build_data_ingestion_dag():
    from airflow import DAG
    from airflow.operators.python import PythonOperator, ShortCircuitOperator
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator

    default_args = {
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    with DAG(
        dag_id="data_ingestion_dag",
        default_args=default_args,
        start_date=datetime(2025, 1, 1),
        schedule="@daily",
        catchup=False,
        tags=["mlops", "ingestion"],
        render_template_as_native_obj=True,
    ) as dag:
        ingest_data = PythonOperator(
            task_id="ingest_data",
            python_callable=ingest_data_task,
        )

        check_drift = PythonOperator(
            task_id="check_drift",
            python_callable=check_drift_task,
        )

        decide_retrain = ShortCircuitOperator(
            task_id="decide_retrain",
            python_callable=decide_retrain_task,
        )

        trigger_training = TriggerDagRunOperator(
            task_id="trigger_training",
            trigger_dag_id="train_candidate_dag",
            wait_for_completion=False,
            conf={
                "dataset_path": str(PROJECT_ROOT / "data" / "raw" / "data_master.csv"),
                "drift_summary": "{{ ti.xcom_pull(task_ids='check_drift', key='drift_summary') }}",
                "triggered_by": "data_ingestion_dag",
            },
        )

        ingest_data >> check_drift >> decide_retrain >> trigger_training

    return dag


dag = build_data_ingestion_dag()
