"""
Production Inference Pipeline

Tries to load from MLflow Model Registry first.
Falls back to local model files when MLflow is unavailable (e.g. Vercel, HuggingFace).
"""

import json
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from predictor.schema import OPTIONAL_FEATURE_DEFAULTS

logger = logging.getLogger(__name__)

LEGACY_MODULE_MAP = {
    "src.features_engineer.preprocessor": "predictor.preprocessor",
    "src.housing_predictor.features.preprocessor": "predictor.preprocessor",
    "housing_predictor.features.preprocessor": "predictor.preprocessor",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "production"


class InferencePipeline:
    """
    Production inference pipeline.

    Usage:
        pipeline = InferencePipeline(model_name="housing_price_predictor", stage="Production")
        predictions = pipeline.predict(new_data_df)
    """

    def __init__(
        self,
        model_name: str = "housing_price_predictor",
        stage: str = "Production",
        version: Optional[Union[int, str]] = None,
        local_model_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.stage = stage
        self.version = str(version) if version is not None else None
        if local_model_dir:
            self.local_model_dir = Path(local_model_dir)
        else:
            self.local_model_dir = LOCAL_MODEL_DIR
        if not self.local_model_dir.is_absolute():
            self.local_model_dir = PROJECT_ROOT / self.local_model_dir
        self._loaded_from = None

        self.model = None
        self.preprocessor = None
        self.metadata = {}
        self.model_version_info = None
        self.client = None

        self._load_artifacts()

    def _load_artifacts(self):
        """Try MLflow first, then S3, then local files."""
        if self._try_load_from_mlflow():
            self._loaded_from = "mlflow"
            logger.info("Loaded model from MLflow registry.")
        elif self._try_load_from_s3():
            self._loaded_from = "s3"
            logger.info("Loaded model from S3 production artifacts.")
        else:
            self._load_from_local()
            self._loaded_from = "local"
            logger.info("Loaded model from local files: %s", self.local_model_dir)

    def _try_load_from_mlflow(self) -> bool:
        """Attempt to load model from MLflow Model Registry."""
        try:
            import mlflow
            import mlflow.sklearn
            from mlflow.tracking import MlflowClient

            self.client = MlflowClient()

            if self.version is not None:
                self.model_version_info = self.client.get_model_version(
                    self.model_name, self.version
                )
            else:
                versions = self.client.search_model_versions(f"name='{self.model_name}'")
                stage_versions = [
                    mv
                    for mv in versions
                    if getattr(mv, "current_stage", None) == self.stage
                ]
                if not stage_versions:
                    logger.warning(
                        "No MLflow model found for '%s' at stage '%s'.",
                        self.model_name,
                        self.stage,
                    )
                    return False
                self.model_version_info = sorted(
                    stage_versions, key=lambda mv: int(mv.version), reverse=True
                )[0]

            model_uri = (
                f"models:/{self.model_name}/{self.version}"
                if self.version
                else f"models:/{self.model_name}/{self.stage}"
            )
            self.model = mlflow.sklearn.load_model(model_uri)
            self.preprocessor = self._download_run_artifact("preprocessor.pkl")
            self.metadata = self._download_run_metadata()
            return True
        except Exception as exc:
            logger.warning("MLflow load failed (%s). Falling back to local files.", exc)
            return False

    def _try_load_from_s3(self) -> bool:
        """Attempt to load production artifacts from S3."""
        bucket = os.getenv("ARTIFACT_BUCKET")
        if not bucket:
            return False

        try:
            import boto3

            s3 = boto3.client(
                "s3",
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            )
            prefix = "models/production"
            s3_base = f"s3://{bucket}/{prefix}"

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                artifact_map = {
                    "model.pkl": tmp_path / "model.pkl",
                    "preprocessor.pkl": tmp_path / "preprocessor.pkl",
                    "metadata.json": tmp_path / "metadata.json",
                    "config.yaml": tmp_path / "config.yaml",
                }

                for artifact_name, local_path in artifact_map.items():
                    s3.download_file(bucket, f"{prefix}/{artifact_name}", str(local_path))

                self.model = _load_pickle_with_compat(artifact_map["model.pkl"])
                self.preprocessor = _load_pickle_with_compat(
                    artifact_map["preprocessor.pkl"]
                )
                with open(artifact_map["metadata.json"], "r") as f:
                    self.metadata = json.load(f)

            self.model_version_info = _LocalVersionStub(Path(s3_base))
            self._loaded_from = "s3"
            logger.info("Loaded production artifacts from %s", s3_base)
            return True
        except Exception as exc:
            logger.warning("S3 load failed (%s). Falling back to local files.", exc)
            return False

    def _download_run_artifact(self, artifact_name: str):
        """Download and deserialise a pickle artifact from an MLflow run."""
        import mlflow

        run_id = self.model_version_info.run_id
        artifact_uri = f"runs:/{run_id}/{artifact_name}"
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
        return _load_pickle_with_compat(Path(local_path))

    def _download_run_metadata(self) -> Dict:
        """Download metadata.json from an MLflow run."""
        import mlflow

        run_id = self.model_version_info.run_id
        artifact_uri = f"runs:/{run_id}/metadata.json"
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
        with open(local_path, "r") as f:
            return json.load(f)

    def _load_from_local(self):
        """Load model, preprocessor, and metadata from local production directory."""
        model_path = self.local_model_dir / "model.pkl"
        preprocessor_path = self.local_model_dir / "preprocessor.pkl"
        metadata_path = self.local_model_dir / "metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Local model not found: {model_path}\n"
                "Make sure models/production/model.pkl exists in the deployment."
            )

        self.model = _load_pickle_with_compat(model_path)
        logger.info("Model loaded from %s", model_path)

        if preprocessor_path.exists():
            self.preprocessor = _load_pickle_with_compat(preprocessor_path)
            logger.info("Preprocessor loaded from %s", preprocessor_path)
        else:
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            logger.warning(
                "metadata.json not found at %s. Using empty dict.", metadata_path
            )
            self.metadata = {}

        self.model_version_info = _LocalVersionStub(self.local_model_dir)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        X = self._prepare_input(X)
        self._validate_input(X)
        X_transformed = self.preprocessor.transform(X)
        predictions = self.model.predict(X_transformed)
        logger.info("Made predictions for %s samples", len(predictions))
        return predictions

    def predict_single(self, features: Dict[str, Union[float, str]]) -> float:
        """Make prediction for a single sample."""
        df = pd.DataFrame([features])
        return float(self.predict(df)[0])

    def predict_with_uncertainty(self, X: pd.DataFrame) -> tuple:
        """Return (predictions, lower_bounds, upper_bounds) using estimator ensemble."""
        preds = self.predict(X)
        X_t = self.preprocessor.transform(self._prepare_input(X))

        interval_cfg = self.metadata.get("prediction_interval") or {}
        if interval_cfg.get("method") == "segmented_relative_error_quantile":
            edges = interval_cfg.get("segment_edges") or []
            rel_q_by_seg = interval_cfg.get("relative_error_quantiles_by_segment")
            rel_q_legacy = interval_cfg.get("relative_error_quantiles") or {}

            pred_nonneg = np.maximum(preds, 0.0)
            q_per_row = None

            if isinstance(rel_q_by_seg, list) and rel_q_by_seg:
                q_arr = np.asarray(rel_q_by_seg, dtype=float)
                edges_arr = np.asarray(edges, dtype=float)
                if len(edges_arr) == len(q_arr) - 1:
                    seg_idx = np.searchsorted(edges_arr, pred_nonneg, side="right")
                    seg_idx = np.clip(seg_idx, 0, len(q_arr) - 1)
                    q_per_row = q_arr[seg_idx]

            if q_per_row is None and len(edges) == 2 and rel_q_legacy:
                edge_1 = float(edges[0])
                edge_2 = float(edges[1])
                q_global = float(rel_q_legacy.get("global", 0.0))
                q_low = float(rel_q_legacy.get("low", q_global))
                q_mid = float(rel_q_legacy.get("mid", q_global))
                q_high = float(rel_q_legacy.get("high", q_global))
                q_per_row = np.where(
                    pred_nonneg < edge_1,
                    q_low,
                    np.where(pred_nonneg < edge_2, q_mid, q_high),
                )

            if q_per_row is not None:
                lower_abs = np.maximum(0.0, pred_nonneg * (1.0 - q_per_row))
                upper_abs = pred_nonneg * (1.0 + q_per_row)
                return preds, lower_abs - preds, upper_abs - preds

        base_estimator = self._unwrap_model_estimator()
        if hasattr(base_estimator, "estimators_"):
            est = base_estimator.estimators_
            trees = est if isinstance(est, list) else np.array(est).ravel().tolist()
            tree_preds = np.array([t.predict(X_t) for t in trees])

            inverse_func = getattr(self.model, "inverse_func", None)
            if callable(inverse_func):
                tree_preds = inverse_func(tree_preds)

            lower = np.percentile(tree_preds, 2.5, axis=0) - preds
            upper = np.percentile(tree_preds, 97.5, axis=0) - preds
            return preds, lower, upper

        logger.warning(
            "Model does not support uncertainty estimation; returning flat bounds."
        )
        zeros = np.zeros_like(preds)
        return preds, zeros, zeros

    def _validate_input(self, X: pd.DataFrame) -> None:
        """Check that all required features are present."""
        if hasattr(self.preprocessor, "numeric_features") and hasattr(
            self.preprocessor, "categorical_features"
        ):
            expected = set(
                self.preprocessor.numeric_features
                + self.preprocessor.categorical_features
            )
        else:
            expected = set(self.metadata.get("feature_names", []))

        missing = expected - set(X.columns)
        if missing:
            raise ValueError(
                f"Missing required features: {missing}\nExpected: {expected}"
            )

    def _prepare_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill optional inference features with stable defaults when omitted."""
        X_prepared = X.copy()
        for feature_name, default_value in OPTIONAL_FEATURE_DEFAULTS.items():
            if feature_name not in X_prepared.columns:
                X_prepared[feature_name] = default_value
            else:
                X_prepared[feature_name] = X_prepared[feature_name].fillna(default_value)
        return X_prepared

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Return a DataFrame of feature importances (model must support it)."""
        base_estimator = self._unwrap_model_estimator()
        if not hasattr(base_estimator, "feature_importances_"):
            raise ValueError("Model does not support feature_importances_.")

        feature_names = self.metadata.get("feature_names", [])
        importances = base_estimator.feature_importances_

        df = pd.DataFrame({"feature": feature_names, "importance": importances})
        return df.sort_values("importance", ascending=False).head(top_n)

    def get_model_info(self) -> Dict:
        """Return basic model information."""
        return {
            "model_name": self.model_name,
            "loaded_from": self._loaded_from,
            "stage": self.stage,
            "metadata": self.metadata,
        }

    def _unwrap_model_estimator(self):
        """Return the inner estimator when a transformed target wrapper is used."""
        if hasattr(self.model, "regressor_"):
            return self.model.regressor_
        if hasattr(self.model, "regressor"):
            return self.model.regressor
        return self.model


class _LocalVersionStub:
    """Minimal stand-in for MLflow ModelVersion when loading locally."""

    def __init__(self, model_dir: Path):
        self.version = "local"
        self.current_stage = "local"
        self.run_id = None
        self.source = str(model_dir)
        self.tags = {}
        self.status = "READY"


class _CompatUnpickler(pickle.Unpickler):
    """Unpickler that remaps legacy module paths to current package paths."""

    def find_class(self, module, name):
        module = LEGACY_MODULE_MAP.get(module, module)
        return super().find_class(module, name)


def _load_pickle_with_compat(path: Path):
    """Load pickle file, retrying with legacy module path remapping when needed."""
    with open(path, "rb") as f:
        prefix = f.read(64)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ValueError(
            f"{path} is a Git LFS pointer, not a real model artifact. "
            "Ensure deployment includes actual .pkl binaries (or load from MLflow)."
        )

    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as exc:
            if "src." not in str(exc) and "housing_predictor" not in str(exc):
                raise
    with open(path, "rb") as f:
        return _CompatUnpickler(f).load()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = InferencePipeline(
        model_name="housing_price_predictor", stage="Production"
    )
    print("Loaded from:", pipeline._loaded_from)
    print("Model info:", pipeline.get_model_info())
