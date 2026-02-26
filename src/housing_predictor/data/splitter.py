"""
Data Splitting Module

Handles train/test/validation splits with proper data leakage prevention.
This is where we answer: "Where should we split the data?"

Key Principle:
- Split EARLY (before any fitting operations)
- Split ONCE (maintain consistency)
- Split RANDOMLY (with fixed seed for reproducibility)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Handles data splitting for ML pipelines.

    Follows the principle:
    1. Split data BEFORE any preprocessing that "learns" from data
    2. Only operations that don't learn (like dropping duplicates) can happen before split
    3. Maintain the same split across experiments for reproducibility

    Usage:
        splitter = DataSplitter(test_size=0.2, val_size=0.1, random_state=42)

        # Option 1: Split and return
        X_train, X_test, X_val, y_train, y_test, y_val = splitter.split(
            X, y, stratify=y
        )

        # Option 2: Split and save
        splitter.split_and_save(
            df,
            target_col='SalePrice',
            output_dir='data/processed/'
        )
    """

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.verbose = verbose

        self._validate_split_sizes()

    def _validate_split_sizes(self) -> None:
        """Validate that split sizes are reasonable"""
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")

        if not 0 <= self.val_size < 1:
            raise ValueError(f"val_size must be between 0 and 1, got {self.val_size}")

        if self.test_size + self.val_size >= 1:
            raise ValueError(
                f"test_size ({self.test_size}) + val_size ({self.val_size}) must be < 1"
            )

    def split(self, X: pd.DataFrame, y: pd.Series, return_val: bool = True) -> Tuple:
        """
        Split data into train/test/validation sets.

        Args:
            X: Feature dataframe
            y: Target series
            stratify: Series to use for stratified splitting (e.g., for classification)
            return_val: Whether to create validation set

        Returns:
            Tuple of (X_train, X_test, X_val, y_train, y_test, y_val)
            If return_val=False: (X_train, X_test, y_train, y_test)
        """
        # First split: Train+Val vs Test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        if self.verbose:
            logger.info(
                f"Initial split - Train+Val: {len(X_temp)}, Test: {len(X_test)}"
            )

        if not return_val:
            if self.verbose:
                logger.info(f"Final split - Train: {len(X_temp)}, Test: {len(X_test)}")
            return X_temp, X_test, y_temp, y_test

        # Second split: Train vs Val (from the Train+Val set)
        # Calculate validation size relative to temp set
        val_size_adjusted = self.val_size / (1 - self.test_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
        )

        if self.verbose:
            logger.info(
                f"Final split - Train: {len(X_train)}, "
                f"Val: {len(X_val)}, Test: {len(X_test)}"
            )
            logger.info(
                f"Split proportions - Train: {len(X_train)/len(X):.2%}, "
                f"Val: {len(X_val)/len(X):.2%}, Test: {len(X_test)/len(X):.2%}"
            )

        return X_train, X_test, X_val, y_train, y_test, y_val

    def split_dataframe(
        self, df: pd.DataFrame, target_col: str, return_val: bool = True
    ) -> Tuple:
        """
        Split dataframe into train/test/validation sets.

        Args:
            df: Input dataframe
            target_col: Name of target column
            stratify_col: Column to use for stratified splitting
            return_val: Whether to create validation set

        Returns:
            Tuple of split dataframes
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]

        return self.split(X, y, return_val=return_val)

    def get_split_statistics(
        self, y_train: pd.Series, y_test: pd.Series, y_val: Optional[pd.Series] = None
    ) -> dict:
        """
        Get statistics about the split to check for data quality.

        Args:
            y_train: Training target
            y_test: Test target
            y_val: Validation target (optional)

        Returns:
            Dictionary of split statistics
        """
        stats = {
            "train": {
                "count": len(y_train),
                "mean": y_train.mean(),
                "std": y_train.std(),
                "min": y_train.min(),
                "max": y_train.max(),
            },
            "test": {
                "count": len(y_test),
                "mean": y_test.mean(),
                "std": y_test.std(),
                "min": y_test.min(),
                "max": y_test.max(),
            },
        }

        if y_val is not None:
            stats["val"] = {
                "count": len(y_val),
                "mean": y_val.mean(),
                "std": y_val.std(),
                "min": y_val.min(),
                "max": y_val.max(),
            }

        return stats


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # import data
    df = pd.read_csv(
        "/Users/hado/Desktop/Career/Coding/Data Engineer/Project/house_price_predictor/data_data/AmesHousing.csv"
    )
    target_col = "SalePrice"
    output_dir = "data/processed/"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # Initialize splitter
    splitter = DataSplitter(test_size=0.2, val_size=0.1, random_state=42)

    X_train, X_test, X_val, y_train, y_test, y_val = splitter.split_dataframe(
        df, target_col="SalePrice"
    )
    # Reconstruct full dataframes
    train_df = X_train.copy()
    train_df[target_col] = y_train

    test_df = X_test.copy()
    test_df[target_col] = y_test

    val_df = X_val.copy()
    val_df[target_col] = y_val

    # Save splits
    train_path = output_path / "train.csv"
    test_path = output_path / "test.csv"
    val_path = output_path / "val.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"\nTrain size: {len(X_train)}")
    print(f"Val size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")

    # Get statistics
    stats = splitter.get_split_statistics(y_train, y_test, y_val)
    print("\nSplit statistics:")
    print(f"Train mean: {stats['train']['mean']:.4f}")
    print(f"Test mean: {stats['test']['mean']:.4f}")
    print(f"Val mean: {stats['val']['mean']:.4f}")
