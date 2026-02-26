"""Data cleaning operations that do not learn from data."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """Safe cleaning operations for pre-split datasets."""

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        original_size = len(df)
        df_clean = df.drop_duplicates()
        removed = original_size - len(df_clean)
        if removed > 0:
            logger.info("Removed %s duplicate rows", removed)
        return df_clean
