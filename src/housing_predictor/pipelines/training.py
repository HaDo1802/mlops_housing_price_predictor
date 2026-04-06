"""Training pipeline orchestration layer."""

from contextlib import nullcontext
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

from housing_predictor.config_manager import ConfigManager
from housing_predictor.data.loader import load_dataframe
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

        self.model = None
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

    def _load_and_clean_data(self) -> pd.DataFrame:
        df = load_dataframe(self.config.data.raw_data_path)

        df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

        if "property_type" in df.columns:
            df = df[~df["property_type"].isin(EXCLUDED_PROPERTY_TYPES)]

        selected = (
            self.config.features.numeric
            + self.config.features.categorical
            + [self.config.data.target_column]
        )
        return df[selected].copy()


    def _remove_outliers(self, X_train, y_train):
        if not self.config.preprocessing.handle_outliers:
            return X_train, y_train

        living = X_train["living_area"].astype(float)
        ppsf = y_train.astype(float) / living.clip(lower=1e-6)

        mask = (
            (living >= self.config.preprocessing.outlier_min_livingarea)
            & (ppsf >= self.config.preprocessing.outlier_ppsf_min)
            & (ppsf <= self.config.preprocessing.outlier_ppsf_max)
        )
        return X_train.loc[mask].copy(), y_train.loc[mask].copy()


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
        # Upweight cheaper homes — a $100k error on a $300k home is worse than on a $3M home.
        """Upweight cheaper homes so the model pays more attention to that segment."""
        y_arr = np.asarray(y, dtype=float)
        median_price = float(np.median(y_arr))
        base = (median_price + 1.0) / (np.maximum(y_arr, 0.0) + 1.0)
        return np.clip(np.sqrt(base), 0.5, 3.0)

    def _evaluate(self, model, X_test_t, y_test) -> tuple[dict, dict]:
        test_pred = np.asarray(model.predict(X_test_t), dtype=float)
        y_test = np.asarray(y_test, dtype=float)

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

        prediction_interval = {
            "method": "segmented_relative_error_quantile",
            "alpha": alpha,
            "coverage": coverage,
            "num_segments": num_segments,
            "segment_edges": edges,
            "relative_error_quantiles_by_segment": q_by_segment,
            "relative_error_quantiles": {"global": global_q},
            "calibration_size": int(len(rel_err)),
        }
        metrics = {
            "test": regression_metrics(y_test, test_pred),
            "validation": None,
        }

        cheap_cutoff = float(np.percentile(y_test, 25))
        cheap_mask = y_test <= cheap_cutoff
        if np.any(cheap_mask):
            metrics["test"]["cheap_segment_mae"] = float(
                np.mean(np.abs(y_test[cheap_mask] - test_pred[cheap_mask]))
            )
            metrics["test"]["cheap_segment_cutoff_p25"] = cheap_cutoff

        return metrics, prediction_interval

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

    def run(self, track: bool = True, promote: bool = True) -> dict:
        self._log_step("START TRAINING RUN")
        if promote and not track:
            raise ValueError("promote=True requires track=True.")

        if track:
            self._ensure_active_experiment()

        run_context = (
            mlflow.start_run(run_name=self.config.training.run_name)
            if track
            else nullcontext()
        )

        with run_context:
            if track:
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
            df_selected = self._load_and_clean_data()

            self._log_step("SPLIT")
            X = df_selected.drop(columns=[self.config.data.target_column])
            y = df_selected[self.config.data.target_column]
            self.X_train, self.X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.data.test_size,
                random_state=self.config.data.random_state,
            )
            logger.info(
                "Final split - Train: %d, Test: %d",
                len(self.X_train), len(self.X_test),
            )
            self.X_train, y_train = self._remove_outliers(self.X_train, y_train)

            self._log_step("PREPROCESS")
            self.X_train_transformed = self.preprocessor.fit_transform(self.X_train)
            X_test_transformed = self.preprocessor.transform(self.X_test)

            self._log_step("TRAIN AND EVALUATE")
            self._apply_monotonic_constraints()
            train_weights = self._compute_price_weights(y_train)
            self.model = self.trainer.fit(
                self.X_train_transformed, y_train, sample_weight=train_weights
            )
            self.metrics, self.prediction_interval = self._evaluate(
                self.model, X_test_transformed, y_test
            )
            logger.info(
                "test_r2=%.4f  test_rmse=%.0f  test_mae=%.0f",
                self.metrics["test"]["r2"],
                self.metrics["test"]["rmse"],
                self.metrics["test"]["mae"],
            )

            if track:
                self._log_step("LOG TO MLFLOW")
                mlflow.log_metrics({
                    "test_r2": self.metrics["test"]["r2"],
                    "test_rmse": self.metrics["test"]["rmse"],
                    "test_mae": self.metrics["test"]["mae"],
                })
                mlflow.sklearn.log_model(
                    self.model, artifact_path="model",
                    signature=mlflow.models.infer_signature(
                        self.X_train_transformed, y_train
                    ),
                )
                mlflow.log_artifact(self.config_path, artifact_path="config")

                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    self.preprocessor.save(str(tmp_path / "preprocessor.pkl"))
                    with open(tmp_path / "metadata.json", "w") as f:
                        json.dump(self._build_metadata(), f, indent=2)
                    mlflow.log_artifacts(str(tmp_path))

            should_promote = False
            if track and promote:
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
                    should_promote = True
                else:
                    self._log_step("SKIP PROMOTION")
                    logger.info("Candidate did not beat production metric.")

            self._log_step("DONE")
            self.metrics["promote_to_production"] = bool(should_promote)
            return self.metrics
