"""Training pipeline orchestration layer."""

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.tracking import MlflowClient

from housing_predictor.config_manager import ConfigManager
from housing_predictor.data.loader import load_dataframe
from housing_predictor.data.splitter import DataSplitter
from housing_predictor.features.preprocessor import ProductionPreprocessor
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
            model_type=self.config.model.model_type,  # now actually used
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

    def load_and_select(self):
        df_raw = load_dataframe(self.config.data.raw_data_path)
        numeric = self.config.features.numeric
        categorical = self.config.features.categorical

        selected = numeric + categorical
        target_col = self.config.data.target_column
        if target_col not in selected:
            selected.append(target_col)

        missing = sorted(set(selected) - set(df_raw.columns))
        if missing:
            raise ValueError(
                f"Configured features not found in input dataframe: {missing}. "
                f"Available columns: {sorted(df_raw.columns.tolist())}"
            )
        self.df_selected = df_raw[selected].copy()

    def split_data(self):
        self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val = (
            self.splitter.split_dataframe(
                self.df_selected, target_col=self.config.data.target_column
            )
        )

    def preprocess_data(self):
        self.X_train_transformed = self.preprocessor.fit_transform(self.X_train)
        self.X_val_transformed = self.preprocessor.transform(self.X_val)
        self.X_test_transformed = self.preprocessor.transform(self.X_test)

    @staticmethod
    def _compute_price_weights(y) -> np.ndarray:
        """
        Upweight cheaper homes so training loss pays more attention to that segment.
        """
        y_arr = np.asarray(y, dtype=float)
        y_nonneg = np.maximum(y_arr, 0.0)
        median_price = float(np.median(y_nonneg))
        base = (median_price + 1.0) / (y_nonneg + 1.0)
        weights = np.sqrt(base)
        return np.clip(weights, 0.5, 3.0)

    def train_and_eval(self):
        train_weights = self._compute_price_weights(self.y_train)
        self.model = self.trainer.fit(
            self.X_train_transformed, self.y_train, sample_weight=train_weights
        )
        test_pred = self.model.predict(self.X_test_transformed)
        val_pred = self.model.predict(self.X_val_transformed)
        y_val = np.asarray(self.y_val, dtype=float)
        pred_val = np.asarray(val_pred, dtype=float)
        pred_val_nonneg = np.maximum(pred_val, 0.0)
        log_residuals = np.log1p(y_val) - np.log1p(pred_val_nonneg)
        alpha = 0.05
        coverage = float(1 - alpha)
        calibration_size = int(len(log_residuals))
        dof = max(calibration_size - 2, 1)
        sum_errs = float(np.sum(np.square(log_residuals)))
        stdev = float(np.sqrt(sum_errs / dof))
        z_score = float(NormalDist().inv_cdf(1 - alpha / 2))
        interval = float(z_score * stdev)
        self.prediction_interval = {
            "method": "lognormal_symmetric_residual_std",
            "alpha": alpha,
            "coverage": coverage,
            "stdev_log_residual": stdev,
            "z_score": z_score,
            "interval_half_width_log": interval,
            "degrees_of_freedom": int(dof),
            "calibration_size": calibration_size,
        }
        self.metrics = {
            "test": regression_metrics(self.y_test, test_pred),
            "validation": regression_metrics(self.y_val, val_pred),
        }
        cheap_cutoff = float(np.percentile(np.asarray(self.y_test, dtype=float), 25))
        cheap_mask = np.asarray(self.y_test, dtype=float) <= cheap_cutoff
        if np.any(cheap_mask):
            cheap_mae = float(
                np.mean(
                    np.abs(
                        np.asarray(self.y_test, dtype=float)[cheap_mask]
                        - test_pred[cheap_mask]
                    )
                )
            )
            self.metrics["test"]["cheap_segment_cutoff_p25"] = cheap_cutoff
            self.metrics["test"]["cheap_segment_mae"] = cheap_mae

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
            "val_size": len(self.X_val),
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
        mlflow.set_experiment(self.config.training.experiment_name)

        with mlflow.start_run(run_name=self.config.training.run_name):
            logger.info("MLflow run started: %s", self.config.training.run_name)
            mlflow.log_params(
                {
                    "test_size": self.config.data.test_size,
                    "val_size": self.config.data.val_size,
                    "random_state": self.config.data.random_state,
                    "scaling_method": self.config.preprocessing.scaling_method,
                    "target_transform": self.config.preprocessing.target_transform,
                    **self.config.model.hyperparameters,
                }
            )
            mlflow.set_tag("model_type", type(self.trainer.get_inner_model()).__name__)
            mlflow.set_tag("model_key", self.config.model.model_type)
            mlflow.set_tag("project", "house_price_prediction")
            mlflow.set_tag("training_date", datetime.now(timezone.utc).isoformat())
            mlflow.set_tag("git_commit", self.registry.get_git_commit())

            self._log_step("LOAD AND SELECT FEATURES")
            self.load_and_select()

            self._log_step("SPLIT DATA")
            self.split_data()

            self._log_step("PREPROCESS DATA")
            self.preprocess_data()

            self._log_step("TRAIN AND EVALUATE")
            self.train_and_eval()
            logger.info(
                "Metrics | test_r2=%.6f test_rmse=%.2f val_r2=%.6f val_rmse=%.2f",
                float(self.metrics["test"]["r2"]),
                float(self.metrics["test"]["rmse"]),
                float(self.metrics["validation"]["r2"]),
                float(self.metrics["validation"]["rmse"]),
            )

            self._log_step("LOG TO MLFLOW")
            mlflow.log_metrics(
                {
                    "test_r2": self.metrics["test"]["r2"],
                    "test_rmse": self.metrics["test"]["rmse"],
                    "test_mae": self.metrics["test"]["mae"],
                    "val_r2": self.metrics["validation"]["r2"],
                    "val_rmse": self.metrics["validation"]["rmse"],
                    "prediction_interval_log_half_width": self.prediction_interval[
                        "interval_half_width_log"
                    ],
                    "test_cheap_segment_mae": self.metrics["test"].get(
                        "cheap_segment_mae", np.nan
                    ),
                }
            )

            mlflow.sklearn.log_model(
                self.model,
                artifact_path="model",
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
                with open(tmp_path / "feature_names.json", "w") as f:
                    json.dump(self.preprocessor.get_feature_names(), f)
                mlflow.log_artifacts(str(tmp_path))

            self._log_step("SAVE LOCAL EXPERIMENT BACKUP")
            backup_path = self.save_artifacts()
            logger.info("Local experiment backup path: %s", backup_path)

            self._log_step("CHECK GATE AGAINST PRODUCTION")
            run_id = mlflow.active_run().info.run_id
            client = MlflowClient()
            gate = check_metric_against_production(
                client=client,
                model_name=self.config.training.registry_model_name,
                candidate_metric=self.metrics["test"]["r2"],
                metric_name="test_r2",
            )
            logger.info("Metric gate result: %s", gate.to_dict())

            if gate.should_register:
                self._log_step("REGISTER + PROMOTE TO PRODUCTION")
                registered_version = self.registry.register_model(
                    run_id=run_id,
                    model_type=type(self.trainer.get_inner_model()).__name__,
                    model_key=self.config.model.model_type,
                )
                promote_result = promote_model_version(
                    client=client,
                    model_name=self.config.training.registry_model_name,
                    version=registered_version,
                    stage="Production",
                )
                local_sync_result = save_local_production_from_objects(
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
                logger.info("Promote result: %s", promote_result)
                logger.info("Local sync result: %s", local_sync_result)
            else:
                self._log_step("SKIP PROMOTION")
                logger.info(
                    "Skipping register/promote: candidate did not beat production."
                )

            self._log_step("END TRAINING RUN")
            self.metrics["promote_to_production"] = bool(gate.should_register)
            return self.metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = TrainingPipeline("conf/config.yaml")
    metrics = pipeline.run()
    print(metrics)
