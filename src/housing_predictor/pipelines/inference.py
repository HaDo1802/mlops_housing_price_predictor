"""
Production Inference Pipeline

Tries to load from MLflow Model Registry first.
Falls back to local model files when MLflow is unavailable (e.g. Vercel, HuggingFace).
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Local artifact directory (used when MLflow is unavailable)
# src/housing_predictor/pipelines/inference.py -> project root is parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "production"


class InferencePipeline:
    """
    Production inference pipeline.

    Usage:
        # Auto-detects MLflow or falls back to local files
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
        self._loaded_from = None  # "mlflow" or "local"

        self.model = None
        self.preprocessor = None
        self.metadata = {}
        self.model_version_info = None
        self.client = None

        self._load_artifacts()

    # ------------------------------------------------------------------
    # Artifact loading — MLflow first, local fallback
    # ------------------------------------------------------------------

    def _load_artifacts(self):
        """Try MLflow first; fall back to local files on any failure."""
        if self._try_load_from_mlflow():
            self._loaded_from = "mlflow"
            logger.info("Loaded model from MLflow registry.")
        else:
            self._load_from_local()
            self._loaded_from = "local"
            logger.info("Loaded model from local files: %s", self.local_model_dir)

    def _try_load_from_mlflow(self) -> bool:
        """Attempt to load model from MLflow Model Registry. Returns True on success."""
        try:
            import mlflow
            import mlflow.sklearn
            from mlflow.tracking import MlflowClient

            self.client = MlflowClient()

            # Resolve version
            if self.version is not None:
                self.model_version_info = self.client.get_model_version(
                    self.model_name, self.version
                )
            else:
                latest = self.client.get_latest_versions(
                    self.model_name, stages=[self.stage]
                )
                if not latest:
                    logger.warning(
                        "No MLflow model found for '%s' at stage '%s'.",
                        self.model_name,
                        self.stage,
                    )
                    return False
                self.model_version_info = sorted(
                    latest, key=lambda mv: int(mv.version), reverse=True
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

    def _download_run_artifact(self, artifact_name: str):
        """Download and deserialise a pickle artifact from an MLflow run."""
        import mlflow

        run_id = self.model_version_info.run_id
        artifact_uri = f"runs:/{run_id}/{artifact_name}"
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
        with open(local_path, "rb") as f:
            return pickle.load(f)

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

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info("Model loaded from %s", model_path)

        if preprocessor_path.exists():
            with open(preprocessor_path, "rb") as f:
                self.preprocessor = pickle.load(f)
            logger.info("Preprocessor loaded from %s", preprocessor_path)
        else:
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            logger.warning("metadata.json not found at %s. Using empty dict.", metadata_path)
            self.metadata = {}

        # Build a minimal version info stub so callers don't break
        self.model_version_info = _LocalVersionStub(self.local_model_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
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
        X_t = self.preprocessor.transform(X)

        if hasattr(self.model, "estimators_"):
            est = self.model.estimators_
            trees = est if isinstance(est, list) else np.array(est).ravel().tolist()
            tree_preds = np.array([t.predict(X_t) for t in trees])
            lower = np.percentile(tree_preds, 2.5, axis=0) - preds
            upper = np.percentile(tree_preds, 97.5, axis=0) - preds
            return preds, lower, upper

        logger.warning("Model does not support uncertainty estimation; returning flat bounds.")
        zeros = np.zeros_like(preds)
        return preds, zeros, zeros

    def _validate_input(self, X: pd.DataFrame) -> None:
        """Check that all required features are present."""
        if hasattr(self.preprocessor, "numeric_features") and hasattr(
            self.preprocessor, "categorical_features"
        ):
            expected = set(
                self.preprocessor.numeric_features + self.preprocessor.categorical_features
            )
        else:
            expected = set(self.metadata.get("feature_names", []))

        missing = expected - set(X.columns)
        if missing:
            raise ValueError(
                f"Missing required features: {missing}\nExpected: {expected}"
            )

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Return a DataFrame of feature importances (model must support it)."""
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model does not support feature_importances_.")

        feature_names = self.metadata.get("feature_names", [])
        importances = self.model.feature_importances_

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

    def list_available_models(self) -> List[Dict]:
        """List registered MLflow versions (only available when loaded from MLflow)."""
        if self._loaded_from != "mlflow" or self.client is None:
            return [{"note": "MLflow not available; model loaded from local files."}]

        versions = self.client.search_model_versions(f"name='{self.model_name}'")
        results = []
        for mv in sorted(versions, key=lambda item: int(item.version)):
            run = self.client.get_run(mv.run_id)
            results.append(
                {
                    "version": mv.version,
                    "stage": mv.current_stage,
                    "run_id": mv.run_id,
                    "status": mv.status,
                    "test_r2": run.data.metrics.get("test_r2"),
                    "val_r2": run.data.metrics.get("val_r2"),
                }
            )
        return results

    def explain_prediction(
        self, features: Dict[str, Union[float, str]], top_n: int = 5
    ) -> Dict:
        """Explain a single prediction with top feature importances."""
        prediction = self.predict_single(features)
        try:
            importance_df = self.get_feature_importance(top_n)
            return {
                "prediction": prediction,
                "top_features": importance_df.to_dict("records"),
            }
        except ValueError:
            return {
                "prediction": prediction,
                "message": "Model does not support feature importance.",
            }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _LocalVersionStub:
    """Minimal stand-in for MLflow ModelVersion when loading locally."""

    def __init__(self, model_dir: Path):
        self.version = "local"
        self.current_stage = "local"
        self.run_id = None
        self.source = str(model_dir)
        self.tags = {}
        self.status = "READY"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = InferencePipeline(model_name="housing_price_predictor", stage="Production")
    print("Loaded from:", pipeline._loaded_from)
    print("Model info:", pipeline.get_model_info())
