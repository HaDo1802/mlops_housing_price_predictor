"""Training pipeline orchestration layer."""

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from housing_predictor.config_manager import ConfigManager
from housing_predictor.data.loader import load_dataframe
from housing_predictor.data.splitter import DataSplitter
from housing_predictor.features.preprocessor import ProductionPreprocessor
from housing_predictor.features.training_schema import (
    DROP_COLUMNS,
    EXCLUDED_PROPERTY_TYPES,
)
from housing_predictor.models.check_metric import check_metric_against_production
from housing_predictor.models.evaluator import regression_metrics
from housing_predictor.models.promote import (
    promote_model_version,
    save_local_production_from_objects,
)
from housing_predictor.models.registry import ModelRegistryManager
from housing_predictor.models.trainer import ModelTrainer

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Thin training orchestrator."""

    def __init__(self, config_path: str = "conf/config.yaml"):
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()

        self.splitter = DataSplitter(
            test_size=self.config.data.test_size,
            val_size=self.config.data.val_size,
            random_state=self.config.data.random_state,
            verbose=True,
        )
        self.preprocessor = ProductionPreprocessor(
            scaling_method=self.config.preprocessing.scaling_method,
            encoding_method=self.config.preprocessing.encoding_method,
            target_transform=self.config.preprocessing.target_transform,
            verbose=True,
        )
        self.trainer = ModelTrainer(
            model_type=self.config.model.model_type,
            hyperparameters=self.config.model.hyperparameters,
            random_state=self.config.model.random_state,
            use_log_target=(self.config.preprocessing.target_transform == "log1p"),
        )
        self.registry = ModelRegistryManager(
            registry_model_name=self.config.training.registry_model_name
        )

        self.metrics = None
        self.prediction_interval = None

    @staticmethod
    def _log_step(step: str) -> None:
        logger.info("")
        logger.info("========== %s ==========", step)

    def _ensure_active_experiment(self) -> None:
        """Restore a soft-deleted MLflow experiment before activating it."""
        experiment_name = self.config.training.experiment_name
        try:
            mlflow.set_experiment(experiment_name)
            return
        except MlflowException as exc:
            if "deleted experiment" not in str(exc).lower():
                raise

        client = MlflowClient()
        experiments = client.search_experiments(
            view_type=ViewType.ALL,
            filter_string=f"name = '{experiment_name}'",
        )
        deleted_experiment = next(
            (exp for exp in experiments if exp.lifecycle_stage == "deleted"),
            None,
        )
        if deleted_experiment is None:
            raise

        logger.warning(
            "MLflow experiment '%s' is deleted. Restoring it automatically.",
            experiment_name,
        )
        client.restore_experiment(deleted_experiment.experiment_id)
        mlflow.set_experiment(experiment_name)

    def load_and_select(self):
        df_raw = load_dataframe(self.config.data.raw_data_path)
        logger.info("Raw data shape: %s", df_raw.shape)

        # ----------------------------------------------------------------
        # 1. Drop columns that are identifiers, 100% null, zero-variance,
        #    or data leakage (price_per_sqft = price / living_area).
        # ----------------------------------------------------------------
        cols_to_drop = [c for c in DROP_COLUMNS if c in df_raw.columns]
        df_raw = df_raw.drop(columns=cols_to_drop)
        logger.info("Dropped %d columns: %s", len(cols_to_drop), cols_to_drop)

        # ----------------------------------------------------------------
        # 2. Exclude property types that are a fundamentally different market
        #    (e.g. MOBILE homes priced off lot-lease, not sqft).
        # ----------------------------------------------------------------
        exclude_types = getattr(
            self.config.preprocessing, "exclude_property_types", EXCLUDED_PROPERTY_TYPES
        )
        if exclude_types and "property_type" in df_raw.columns:
            before = len(df_raw)
            df_raw = df_raw[~df_raw["property_type"].isin(exclude_types)]
            logger.info(
                "Excluded property types %s: removed %d rows.",
                exclude_types, before - len(df_raw),
            )

        # ----------------------------------------------------------------
        # 3. Select only the configured feature columns + target.
        # ----------------------------------------------------------------
        numeric = self.config.features.numeric
        categorical = self.config.features.categorical
        target_col = self.config.data.target_column

        selected = numeric + categorical + [target_col]
        missing_cols = sorted(set(selected) - set(df_raw.columns))
        if missing_cols:
            raise ValueError(
                f"Configured features not found in dataframe: {missing_cols}. "
                f"Available: {sorted(df_raw.columns.tolist())}"
            )

        self.df_selected = df_raw[selected].copy()
        logger.info("Selected %d rows, %d columns.", *self.df_selected.shape)

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = (
            self.splitter.split_dataframe(
                self.df_selected,
                target_col=self.config.data.target_column,
                return_val=False,
            )
        )
        self.X_val = None
        self.y_val = None
        self._remove_training_outliers()

    def _remove_training_outliers(self) -> None:
        """
        Remove outliers from training split only — never touches test data.

        Strategy: filter on price-per-sqft band rather than raw IQR on price.
        This catches data errors (price = $2k) and ultra-luxury outliers
        (ppsf > $800) without arbitrarily capping the price range.
        """
        cfg = self.config.preprocessing
        if not getattr(cfg, "handle_outliers", False):
            return

        method = str(getattr(cfg, "outlier_method", "iqr")).lower()
        initial_count = len(self.y_train)

        if method == "ppsf_band":
            if "living_area" not in self.X_train.columns:
                logger.warning("ppsf_band filter skipped: 'living_area' not in features.")
                return

            y = self.y_train.astype(float).to_numpy()
            living = self.X_train["living_area"].astype(float).to_numpy()

            ppsf_min = float(getattr(cfg, "outlier_ppsf_min", 80.0))
            ppsf_max = float(getattr(cfg, "outlier_ppsf_max", 800.0))
            min_living = float(getattr(cfg, "outlier_min_livingarea", 400.0))

            ppsf = np.divide(y, np.maximum(living, 1e-6))
            keep_mask = (
                (living >= min_living)
                & (ppsf >= ppsf_min)
                & (ppsf <= ppsf_max)
            )
            removed = initial_count - int(keep_mask.sum())
            logger.info(
                "Outlier filter (ppsf_band): ppsf=[%.0f, %.0f] living_area>=%.0f "
                "→ removed %d / %d rows.",
                ppsf_min, ppsf_max, min_living, removed, initial_count,
            )

        elif method == "iqr":
            y = self.y_train.astype(float)
            q1, q3 = float(y.quantile(0.25)), float(y.quantile(0.75))
            iqr = q3 - q1
            mult = float(getattr(cfg, "outlier_iqr_multiplier", 1.5))
            lower, upper = q1 - mult * iqr, q3 + mult * iqr
            keep_mask = (y >= lower).to_numpy() & (y <= upper).to_numpy()
            removed = initial_count - int(keep_mask.sum())
            logger.info(
                "Outlier filter (iqr): [%.0f, %.0f] → removed %d / %d rows.",
                lower, upper, removed, initial_count,
            )

        else:
            logger.warning("Unknown outlier_method '%s'. Skipping.", method)
            return

        kept = int(keep_mask.sum())
        if kept < 30:
            logger.warning(
                "Filter would leave only %d rows — skipping to avoid underfitting.", kept
            )
            return

        self.X_train = self.X_train.loc[keep_mask].copy()
        self.y_train = self.y_train.loc[keep_mask].copy()

    def preprocess_data(self):
        self.X_train_transformed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_transformed = self.preprocessor.transform(self.X_test)

    def _apply_monotonic_constraints(self) -> None:
        """Enforce monotonic increase for key capacity features in HGB."""
        if self.config.model.model_type != "hist_gradient_boosting":
            return

        n_features = int(self.X_train_transformed.shape[1])
        constraints = [0] * n_features
        numeric = list(getattr(self.preprocessor, "numeric_features", []))
        for feature_name in ("bedrooms", "bathrooms", "living_area"):
            if feature_name in numeric:
                constraints[numeric.index(feature_name)] = 1

        self.trainer.set_hyperparameter("monotonic_cst", constraints)
        logger.info("Applied monotonic constraints for bedrooms, bathrooms, living_area.")

    @staticmethod
    def _compute_price_weights(y) -> np.ndarray:
        """Upweight cheaper homes so the model pays more attention to that segment."""
        y_arr = np.asarray(y, dtype=float)
        median_price = float(np.median(y_arr))
        base = (median_price + 1.0) / (np.maximum(y_arr, 0.0) + 1.0)
        return np.clip(np.sqrt(base), 0.5, 3.0)

    def train_and_eval(self):
        self._apply_monotonic_constraints()
        train_weights = self._compute_price_weights(self.y_train)
        self.model = self.trainer.fit(
            self.X_train_transformed, self.y_train, sample_weight=train_weights
        )

        test_pred = np.asarray(self.model.predict(self.X_test_transformed), dtype=float)
        y_test = np.asarray(self.y_test, dtype=float)

        # Segmented relative-error calibration for prediction intervals
        alpha = 0.05
        coverage = float(1 - alpha)
        pred_nonneg = np.maximum(test_pred, 0.0)
        rel_err = np.abs(y_test - pred_nonneg) / np.maximum(np.abs(y_test), 1.0)
        global_q = float(np.quantile(rel_err, 1 - alpha, method="higher"))

        num_segments = max(2, int(self.config.preprocessing.interval_num_segments))
        min_seg_size = max(1, int(self.config.preprocessing.interval_min_segment_size))
        probs = np.linspace(0, 1, num_segments + 1)[1:-1]
        edges = (
            np.quantile(pred_nonneg, probs).astype(float).tolist() if len(probs) else []
        )
        edges = np.maximum.accumulate(np.asarray(edges, dtype=float)).tolist()

        q_by_segment = []
        for seg_idx in range(num_segments):
            if seg_idx == 0:
                mask = pred_nonneg < edges[0] if edges else np.ones(len(pred_nonneg), dtype=bool)
            elif seg_idx == num_segments - 1:
                mask = pred_nonneg >= edges[-1] if edges else np.ones(len(pred_nonneg), dtype=bool)
            else:
                mask = (pred_nonneg >= edges[seg_idx - 1]) & (pred_nonneg < edges[seg_idx])
            seg_err = rel_err[mask]
            q_by_segment.append(
                float(np.quantile(seg_err, 1 - alpha, method="higher"))
                if seg_err.size >= min_seg_size else global_q
            )

        self.prediction_interval = {
            "method": "segmented_relative_error_quantile",
            "alpha": alpha,
            "coverage": coverage,
            "num_segments": num_segments,
            "segment_edges": edges,
            "relative_error_quantiles_by_segment": q_by_segment,
            "relative_error_quantiles": {"global": global_q},
            "calibration_size": int(len(rel_err)),
        }
        self.metrics = {
            "test": regression_metrics(self.y_test, test_pred),
            "validation": None,
        }

        cheap_cutoff = float(np.percentile(y_test, 25))
        cheap_mask = y_test <= cheap_cutoff
        if np.any(cheap_mask):
            self.metrics["test"]["cheap_segment_mae"] = float(
                np.mean(np.abs(y_test[cheap_mask] - test_pred[cheap_mask]))
            )
            self.metrics["test"]["cheap_segment_cutoff_p25"] = cheap_cutoff

    def _build_metadata(self) -> dict:
        inner_model = self.trainer.get_inner_model()
        return {
            "model_type": type(inner_model).__name__,
            "model_key": self.config.model.model_type,
            "target_transform": self.config.preprocessing.target_transform,
            "hyperparameters": self.config.model.hyperparameters,
            "test_metrics": self.metrics["test"],
            "val_metrics": self.metrics["validation"],
            "feature_names": self.preprocessor.get_feature_names(),
            "prediction_interval": self.prediction_interval,
            "train_size": len(self.X_train),
            "val_size": 0,
            "test_size": len(self.X_test),
        }

    def save_artifacts(self, output_dir: str = None) -> Path:
        if output_dir is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = Path("models/experiments") / timestamp
        else:
            output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        import pickle
        with open(output_path / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)
        self.preprocessor.save(str(output_path / "preprocessor.pkl"))
        self.config_manager.save_config(str(output_path / "config.yaml"))
        with open(output_path / "metadata.json", "w") as f:
            json.dump(self._build_metadata(), f, indent=2)
        return output_path

    def run(self) -> dict:
        self._log_step("START TRAINING RUN")
        self._ensure_active_experiment()

        with mlflow.start_run(run_name=self.config.training.run_name):
            mlflow.log_params({
                "test_size": self.config.data.test_size,
                "random_state": self.config.data.random_state,
                "scaling_method": self.config.preprocessing.scaling_method,
                "target_transform": self.config.preprocessing.target_transform,
                "outlier_method": self.config.preprocessing.outlier_method,
                **self.config.model.hyperparameters,
            })
            mlflow.set_tag("model_type", type(self.trainer.get_inner_model()).__name__)
            mlflow.set_tag("model_key", self.config.model.model_type)
            mlflow.set_tag("training_date", datetime.now(timezone.utc).isoformat())
            mlflow.set_tag("git_commit", self.registry.get_git_commit())

            self._log_step("LOAD AND SELECT")
            self.load_and_select()

            self._log_step("SPLIT")
            self.split_data()

            self._log_step("PREPROCESS")
            self.preprocess_data()

            self._log_step("TRAIN AND EVALUATE")
            self.train_and_eval()
            logger.info(
                "test_r2=%.4f  test_rmse=%.0f  test_mae=%.0f",
                self.metrics["test"]["r2"],
                self.metrics["test"]["rmse"],
                self.metrics["test"]["mae"],
            )

            self._log_step("LOG TO MLFLOW")
            mlflow.log_metrics({
                "test_r2": self.metrics["test"]["r2"],
                "test_rmse": self.metrics["test"]["rmse"],
                "test_mae": self.metrics["test"]["mae"],
            })
            mlflow.sklearn.log_model(
                self.model, artifact_path="model",
                signature=mlflow.models.infer_signature(
                    self.X_train_transformed, self.y_train
                ),
            )
            mlflow.log_artifact(self.config_path, artifact_path="config")

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                self.preprocessor.save(str(tmp_path / "preprocessor.pkl"))
                with open(tmp_path / "metadata.json", "w") as f:
                    json.dump(self._build_metadata(), f, indent=2)
                mlflow.log_artifacts(str(tmp_path))

            self._log_step("METRIC GATE")
            run_id = mlflow.active_run().info.run_id
            client = MlflowClient()
            gate = check_metric_against_production(
                client=client,
                model_name=self.config.training.registry_model_name,
                candidate_metric=self.metrics["test"]["r2"],
                metric_name="test_r2",
            )
            logger.info("Gate result: %s", gate.to_dict())

            if gate.should_register:
                self._log_step("REGISTER + PROMOTE")
                version = self.registry.register_model(
                    run_id=run_id,
                    model_type=type(self.trainer.get_inner_model()).__name__,
                    model_key=self.config.model.model_type,
                )
                promote_model_version(
                    client=client,
                    model_name=self.config.training.registry_model_name,
                    version=version,
                    stage="Production",
                )
                save_local_production_from_objects(
                    model=self.model,
                    preprocessor=self.preprocessor,
                    metadata=self._build_metadata(),
                    config_dict={
                        "data": self.config.data.__dict__,
                        "features": self.config.features.__dict__,
                        "preprocessing": self.config.preprocessing.__dict__,
                        "model": self.config.model.__dict__,
                        "training": self.config.training.__dict__,
                    },
                )
            else:
                self._log_step("SKIP PROMOTION")
                logger.info("Candidate did not beat production metric.")

            self._log_step("DONE")
            self.metrics["promote_to_production"] = bool(gate.should_register)
            return self.metrics
