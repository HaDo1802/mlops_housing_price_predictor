import os
import logging
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _db_conn_kwargs() -> dict:
    return {
        "host": os.getenv("SUPABASE_DB_HOST"),
        "port": os.getenv("SUPABASE_DB_PORT", "5432"),
        "dbname": os.getenv("SUPABASE_DB_NAME", "postgres"),
        "user": os.getenv("SUPABASE_DB_USER"),
        "password": os.getenv("SUPABASE_DB_PASSWORD"),
        "sslmode": os.getenv("SUPABASE_DB_SSLMODE", "require"),
    }


def _validate_db_env() -> None:
    conn_kwargs = _db_conn_kwargs()
    missing = [k for k in ("host", "user", "password") if not conn_kwargs.get(k)]
    if missing:
        logger.error("Missing required Supabase DB environment variables: %s", missing)
        raise RuntimeError(
            "Missing Supabase DB env vars: "
            + ", ".join(f"SUPABASE_DB_{m.upper()}" for m in missing)
        )
    logger.info("Supabase DB env validation passed.")


def _run_sql(query: str) -> pd.DataFrame:
    _validate_db_env()
    logger.info("Executing query against Supabase.")
    # Prefer SQLAlchemy engine to avoid pandas DBAPI2 warning.
    try:
        from sqlalchemy import create_engine

        cfg = _db_conn_kwargs()
        db_url = (
            f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}"
            f"@{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
            f"?sslmode={cfg['sslmode']}"
        )
        engine = create_engine(db_url)
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn)
            logger.info("Query completed via SQLAlchemy. Rows fetched: %d", len(df))
            return df
    except Exception as exc:
        logger.warning(
            "SQLAlchemy path failed (%s). Falling back to psycopg2 connection.", exc
        )
        # Fallback keeps script functional if SQLAlchemy is unavailable.
        with psycopg2.connect(**_db_conn_kwargs()) as conn:
            df = pd.read_sql_query(query, conn)
            logger.info(
                "Query completed via psycopg2 fallback. Rows fetched: %d", len(df)
            )
            return df


def load_dashboard_df() -> pd.DataFrame:
    query = """
        select
            f.price,
            f.bedrooms,
            f.bathrooms,
            f.living_area as livingarea,
            d.property_type as propertytype,
            f.listing_status as listingstatus,
            d.vegas_district,
            d.latitude,
            d.longitude,
            f.snapshot_date
        from gold.fact_property_latest f
        inner join gold.dim_property d
            on d.property_id = f.property_id
        left join gold.dim_date dd
            on dd.date_day = f.snapshot_date
    """
    return _run_sql(query)


if __name__ == "__main__":
    logger.info("Starting Supabase extract job.")
    data = load_dashboard_df()
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / "raw" / "data_master.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving dataset to %s", output_path)
    data.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(data), output_path)
