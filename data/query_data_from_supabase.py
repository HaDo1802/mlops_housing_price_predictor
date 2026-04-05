import os
import logging
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
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
    cfg = _db_conn_kwargs()
    missing = [k for k in ("host", "user", "password") if not cfg.get(k)]
    if missing:
        raise RuntimeError(
            "Missing Supabase DB env vars: "
            + ", ".join(f"SUPABASE_DB_{m.upper()}" for m in missing)
        )
    logger.info("Supabase DB env validation passed.")


def _get_engine():
    _validate_db_env()
    cfg = _db_conn_kwargs()
    db_url = URL.create(
        drivername="postgresql+psycopg2",
        username=cfg["user"],
        password=cfg["password"],
        host=cfg["host"],
        port=int(cfg["port"]),
        database=cfg["dbname"],
        query={"sslmode": cfg["sslmode"]},
    )
    return create_engine(db_url)


def _run_sql(query: str, params: dict | None = None) -> pd.DataFrame:
    """
    Run a SQL query and return a DataFrame.
    Use :key placeholders in query string, e.g. WHERE date = :snapshot_date.
    """
    logger.info("Executing query against Supabase.")
    engine = _get_engine()
    with engine.connect() as conn:
        sql = text(query)
        if params:
            sql = sql.bindparams(**params)
        df = pd.read_sql_query(sql, conn)
        logger.info("Query completed. Rows fetched: %d", len(df))
        return df


def load_dashboard_df() -> pd.DataFrame:
    """
    Load the latest snapshot of every property from gold.mart_property_current.

    This view is deduplicated — one row per property_id reflecting its most
    recent values. No time-series joins needed.

    Columns: property_id, snapshot_date, street_address, city, state,
             zip_code, vegas_district, latitude, longitude, property_type,
             price, zestimate, rentzestimate, bedrooms, bathrooms,
             living_area, normalized_lot_area_value, normalized_lot_area_unit,
             days_on_zillow, listing_status, price_per_sqft
    """
    query = """
        SELECT *
        FROM gold.mart_property_current
        WHERE price IS NOT NULL
          AND price > 0
    """
    logger.info("Loading gold.mart_property_current ...")
    return _run_sql(query)


if __name__ == "__main__":
    logger.info("Starting Supabase extract job.")
    df = load_dashboard_df()

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / "raw" / "data_master.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)
