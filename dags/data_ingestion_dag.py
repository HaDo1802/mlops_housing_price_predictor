"""Daily data ingestion DAG with drift gate and conditional retraining trigger.

This DAG ingests latest housing data, runs a temporary drift bypass gate,
and conditionally triggers the retraining DAG.

Drift logic is intentionally bypassed while the dataset is small.
To re-enable true drift checks, uncomment the disabled PSI block in
`check_drift_task` once the dataset exceeds ~5000 rows.
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


def ingest_data_task(**context) -> None:
    from airflow.exceptions import AirflowSkipException
    from data.query_data_from_supabase import load_dashboard_df

    required_env_vars = [
        "SUPABASE_DB_HOST",
        "SUPABASE_DB_USER",
        "SUPABASE_DB_PASSWORD",
    ]
    missing = [name for name in required_env_vars if not os.getenv(name)]
    if missing:
        raise AirflowSkipException(
            "Skipping ingestion because required Supabase env vars are missing: "
            + ", ".join(missing)
        )

    output_path = PROJECT_ROOT / "data" / "raw" / "data_master.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_dashboard_df()
    df.to_csv(output_path, index=False)

    row_count = int(len(df))
    logging.info("Ingestion complete. Saved %s rows to %s", row_count, output_path)
    context["ti"].xcom_push(key="row_count", value=row_count)


def check_drift_task(**context) -> None:
    import pandas as pd

    # df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "data_master.csv")

    # ----------------------------------------------------------------
    # DRIFT LOGIC DISABLED — dataset too small for PSI to be reliable.
    # Uncomment this block once the dataset grows beyond ~5000 rows.
    # ----------------------------------------------------------------
    # result = run_drift_check(
    #     current_df=df,
    #     reference_path=Path(DEFAULT_REFERENCE_PATH),
    #     psi_threshold=PSI_THRESHOLD_MODERATE,
    # )
    # should_retrain = result["should_retrain"]
    # drift_summary  = result
    # ----------------------------------------------------------------

    should_retrain = True
    drift_summary = {
        "psi": None,
        "drift_level": "bypassed",
        "should_retrain": True,
        "reason": "Drift detection disabled — forced trigger until dataset is large enough.",
    }

    logging.warning(
        "Drift check bypassed — always triggering retraining until dataset is large enough for PSI to be reliable."
    )

    context["ti"].xcom_push(key="should_retrain", value=should_retrain)
    context["ti"].xcom_push(key="drift_summary", value=drift_summary)


def decide_retrain_task(**context) -> bool:
    should_retrain = context["ti"].xcom_pull(
        task_ids="check_drift", key="should_retrain"
    )
    return should_retrain


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
        tags=["mlops", "housing_predictor", "ingestion"],
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

        trigger_retraining = TriggerDagRunOperator(
            task_id="trigger_retraining",
            trigger_dag_id="retraining_dag",
            wait_for_completion=False,
            reset_dag_run=True,
            conf={
                "dataset_path": str(PROJECT_ROOT / "data" / "raw" / "data_master.csv"),
                "drift_summary": "{{ ti.xcom_pull(task_ids='check_drift', key='drift_summary') }}",
                "triggered_by": "data_ingestion_dag",
            },
        )

        ingest_data >> check_drift >> decide_retrain >> trigger_retraining

    return dag


dag = build_data_ingestion_dag()
