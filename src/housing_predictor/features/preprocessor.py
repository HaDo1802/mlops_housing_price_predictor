"""
Production-Grade Data Preprocessing Module

Key principles:
1. Fit ONLY on training data
2. Transform train/val/test using the SAME fitted transformers
3. Save fitted transformers for production use
4. Handle edge cases (unknown categories, missing features, etc.)

This module answers: "How do we preprocess to avoid data leakage?"
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from housing_predictor.features.training_schema import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


logger = logging.getLogger(__name__)


class ProductionPreprocessor:
    """
    Production-grade preprocessor that prevents data leakage.

    CRITICAL RULES:
    1. fit_transform() - Use ONLY on training data
    2. transform() - Use on validation, test, and production data
    3. Always save fitted preprocessor for production

    Usage:
        # Training
        preprocessor = ProductionPreprocessor()
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_val_transformed = preprocessor.transform(X_val)
        X_test_transformed = preprocessor.transform(X_test)
        preprocessor.save('models/preprocessor.pkl')

        # Production (weeks/months later)
        preprocessor = ProductionPreprocessor.load('models/preprocessor.pkl')
        X_new_transformed = preprocessor.transform(X_new)
    """

    def __init__(
        self,
        scaling_method: str = "standard",
        encoding_method: str = "onehot",
        handle_unknown: str = "ignore",
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        target_transform: str = "log1p",
        verbose: bool = True,
    ):
        self.scaling_method = scaling_method
        self.encoding_method = encoding_method
        self.handle_unknown = handle_unknown
        self.target_transform = target_transform
        self.verbose = verbose

        # These will be fitted on training data
        self.preprocessor_pipeline = None
        self.numeric_features = list(numeric_features or NUMERIC_FEATURES)
        self.categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
        self.expected_features = self.numeric_features + self.categorical_features
        self.feature_names_out = None
        self.is_fitted = False

    def _identify_feature_types(self, X: pd.DataFrame) -> None:
        """Validate and lock expected numeric and categorical feature groups."""
        missing_features = set(self.expected_features) - set(X.columns)
        if missing_features:
            raise ValueError(
                f"Missing required features in input data: {sorted(missing_features)}. "
                f"Expected model features: {sorted(self.expected_features)}"
            )

        if self.verbose:
            logger.info(f"Using {len(self.numeric_features)} numeric features")
            logger.info(f"Using {len(self.categorical_features)} categorical features")

    def _get_scaler(self):
        """Return the configured scaler."""
        if self.scaling_method == "standard":
            return StandardScaler()
        elif self.scaling_method == "minmax":
            return MinMaxScaler()
        elif self.scaling_method == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")

    def _get_encoder(self):
        """Return the configured encoder."""
        if self.encoding_method == "onehot":
            return OneHotEncoder(
                sparse_output=False,
                handle_unknown=self.handle_unknown,
                drop=None,  # Keep all categories for explainability
            )
        else:
            raise ValueError(f"Unknown encoding method: {self.encoding_method}")

    def _get_imputer(self, strategy: str):
        """Return the configured imputer."""
        return SimpleImputer(strategy=strategy)

    def _build_preprocessor(self) -> ColumnTransformer:
        """Build the preprocessing pipeline."""
        transformers = []

        # Add numeric transformer
        if self.numeric_features:
            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", self._get_imputer("median")),
                    ("scaler", self._get_scaler()),
                ]
            )
            transformers.append(("numeric", numeric_transformer, self.numeric_features))

        # Add categorical transformer
        if self.categorical_features:
            categorical_transformer = Pipeline(steps=[("encoder", self._get_encoder())])
            transformers.append(
                ("categorical", categorical_transformer, self.categorical_features)
            )

        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",  # Drop any other columns
            verbose=self.verbose,
        )

        return preprocessor

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit the preprocessor on training data and transform it."""
        if self.is_fitted:
            logger.warning(
                "Preprocessor is already fitted! "
                "Are you sure you want to refit? This should only happen on training data."
            )

        # Identify feature types
        self._identify_feature_types(X)

        # Build preprocessing pipeline
        self.preprocessor_pipeline = self._build_preprocessor()

        # Fit and transform
        if self.verbose:
            logger.info("Fitting preprocessor on training data...")

        X_transformed = self.preprocessor_pipeline.fit_transform(X)

        # Store feature names after transformation
        self._store_feature_names()

        self.is_fitted = True

        if self.verbose:
            logger.info(f"Preprocessing complete. Output shape: {X_transformed.shape}")

        return X_transformed

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using the fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError(
                "Preprocessor not fitted yet! Call fit_transform() on training data first."
            )

        # Validate that input has expected features
        self._validate_input_features(X)

        # Transform
        X_transformed = self.preprocessor_pipeline.transform(X)

        if self.verbose:
            logger.info(
                f"Transformed {len(X)} samples. Output shape: {X_transformed.shape}"
            )

        return X_transformed

    def _validate_input_features(self, X: pd.DataFrame) -> None:
        """Validate that input has the same features as training data."""
        expected_features = set(self.expected_features)
        actual_features = set(X.columns)

        missing_features = expected_features - actual_features
        extra_features = actual_features - expected_features

        if missing_features:
            raise ValueError(
                f"Missing features in input data: {missing_features}\n"
                f"Expected features: {expected_features}"
            )

        if extra_features and self.verbose:
            logger.warning(
                f"Extra features in input data will be ignored: {extra_features}"
            )

    def _store_feature_names(self) -> None:
        """Store feature names after transformation."""
        feature_names = []

        # Get numeric feature names
        if self.numeric_features:
            feature_names.extend(self.numeric_features)

        # Get encoded categorical feature names
        if self.categorical_features:
            encoder = self.preprocessor_pipeline.named_transformers_[
                "categorical"
            ].named_steps["encoder"]
            cat_feature_names = encoder.get_feature_names_out(self.categorical_features)
            feature_names.extend(cat_feature_names.tolist())

        self.feature_names_out = feature_names

    def get_feature_names(self) -> List[str]:
        """Return feature names after transformation."""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted yet!")

        return self.feature_names_out

    def save(self, filepath: str) -> None:
        """Save the fitted preprocessor to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor!")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

        if self.verbose:
            logger.info(f"Preprocessor saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> "ProductionPreprocessor":
        """Load a fitted preprocessor from disk."""
        with open(filepath, "rb") as f:
            preprocessor = pickle.load(f)

        if not preprocessor.is_fitted:
            logger.warning("Loaded preprocessor is not fitted!")

        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor
