"""Data ingestion and dataset cleaning services."""

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from predictor.config import MLConfig
from predictor.schema import DROP_COLUMNS, NUMERIC_FEATURES

load_dotenv()

logger = logging.getLogger(__name__)


class DataIngestor:
    """Fetch raw data and apply business/statistical cleaning rules."""

    def __init__(self, config: MLConfig):
        self.config = config

    def fetch_data(self) -> pd.DataFrame:
        """Load the raw dataset from query Supabase."""
        logger.info("Local data file not found. Querying Supabase view instead.")
        host = os.getenv("SUPABASE_DB_HOST")
        user = os.getenv("SUPABASE_DB_USER")
        password = os.getenv("SUPABASE_DB_PASSWORD")
        if not host or not user or not password:
            raise RuntimeError(
                "Missing Supabase DB env vars: "
                "SUPABASE_DB_HOST, SUPABASE_DB_USER, SUPABASE_DB_PASSWORD"
            )

        engine = create_engine(
            URL.create(
                drivername="postgresql+psycopg2",
                username=user,
                password=password,
                host=host,
                port=int(os.getenv("SUPABASE_DB_PORT", "5432")),
                database=os.getenv("SUPABASE_DB_NAME", "postgres"),
                query={"sslmode": os.getenv("SUPABASE_DB_SSLMODE", "require")},
            )
        )
        query = """
            SELECT *
            FROM gold.mart_property_current
            WHERE price IS NOT NULL
        """
        with engine.connect() as conn:
            return pd.read_sql_query(text(query), conn)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply business cleaning rules to the raw dataset."""
        cleaned = df.drop_duplicates().copy()
        cleaned = cleaned.drop(columns=[c for c in DROP_COLUMNS if c in cleaned.columns])

        exclude_types = self.config.preprocessing.exclude_property_types
        if exclude_types and "property_type" in cleaned.columns:
            cleaned = cleaned[~cleaned["property_type"].isin(exclude_types)]

        numeric_cols = [col for col in NUMERIC_FEATURES if col in cleaned.columns]
        if numeric_cols:
            cleaned.loc[:, numeric_cols] = cleaned[numeric_cols].astype("float64")

        return cleaned

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply business-rule thresholds for price and living area."""
        if not self.config.preprocessing.handle_outliers:
            return df
        target_col = self.config.data.target_column
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' is missing from dataset.")
        if "living_area" not in df.columns:
            logger.warning("Outlier filter skipped: 'living_area' not in dataset.")
            return df

        initial_count = len(df)
        keep_mask = (
            (df[target_col].astype(float) >= self.config.preprocessing.min_price)
            & (df[target_col].astype(float) <= self.config.preprocessing.max_price)
            & (df["living_area"].astype(float) >= self.config.preprocessing.min_living_area)
        )
        filtered = df.loc[keep_mask].copy()
        logger.info(
            "Removed %d rows using business-rule thresholds.",
            initial_count - len(filtered),
        )
        return filtered
