"""Feature preprocessing helpers."""

from pathlib import Path
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predictor.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class ProductionPreprocessor:
    """Fit and apply numeric/categorical feature transforms."""

    def __init__(
        self,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        scaling_method: str = "standard",
        encoding_method: str = "onehot",
        target_transform: str = "log1p",
        verbose: bool = True,
    ):
        self.numeric_features = list(numeric_features or NUMERIC_FEATURES)
        self.categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
        self.scaling_method = scaling_method
        self.encoding_method = encoding_method
        self.target_transform = target_transform
        self.verbose = verbose
        self.is_fitted = False

        self.pipeline = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        [
                            ("num_imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    self.numeric_features,
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            ("cat_imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    sparse_output=False,
                                ),
                            ),
                        ]
                    ),
                    self.categorical_features,
                ),
            ],
            remainder="drop",
            verbose=verbose,
        )

    def fit_transform(self, X: pd.DataFrame):
        missing = set(self.numeric_features + self.categorical_features) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")
        X_out = self.pipeline.fit_transform(X)
        self.is_fitted = True
        return X_out

    def transform(self, X: pd.DataFrame):
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")
        missing = set(self.numeric_features + self.categorical_features) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")
        return self.pipeline.transform(X)

    def get_feature_names(self) -> list[str]:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before reading feature names.")

        feature_names = list(self.numeric_features)
        encoder = self.pipeline.named_transformers_["categorical"].named_steps["encoder"]
        feature_names.extend(encoder.get_feature_names_out(self.categorical_features).tolist())
        return feature_names

    def save(self, path: str | Path):
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path):
        with open(Path(path), "rb") as f:
            return pickle.load(f)
