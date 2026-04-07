"""Training pipeline orchestration layer."""

from contextlib import nullcontext
import json
import logging
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

from predictor.config import ConfigManager, MLConfig
from predictor.data_ingest import DataIngestor
from predictor.models import TrainerFactory
from predictor.preprocessor import ProductionPreprocessor
from predictor.registry import ModelRegistryManager, save_local_production_from_objects
from predictor.schema import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES
from predictor.utils import evaluate_predictions

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrate ingestion, preprocessing, training, evaluation, and promotion."""

    def __init__(self, config_path: str = "conf/config.yaml"):
        self.config_path = Path(config_path)
        self.config_manager = ConfigManager(str(self.config_path))
        self.config: MLConfig = self.config_manager.get_config()

        self.data_ingestor = DataIngestor(self.config)
        self.preprocessor = ProductionPreprocessor(
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            scaling_method=self.config.preprocessing.scaling_method,
            encoding_method=self.config.preprocessing.encoding_method,
            target_transform=self.config.preprocessing.target_transform,
            verbose=True,
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

    def _select_training_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        selected = list(MODEL_FEATURES) + [self.config.data.target_column]
        missing_cols = sorted(set(selected) - set(df.columns))
        if missing_cols:
            raise ValueError(
                f"Configured features not found in dataframe: {missing_cols}. "
                f"Available: {sorted(df.columns.tolist())}"
            )
        return df[selected].copy()

    def _build_metadata(self) -> dict:
        inner_model = TrainerFactory.get_inner_model(self.model)
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

    def save_artifacts(self, output_dir: str | Path | None = None) -> Path:
        output_path = Path(output_dir) if output_dir else Path("models/experiments")
        output_path.mkdir(parents=True, exist_ok=True)

        import pickle

        with open(output_path / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)
        self.preprocessor.save(output_path / "preprocessor.pkl")
        self.config_manager.save_config(str(output_path / "config.yaml"))
        with open(output_path / "metadata.json", "w") as f:
            json.dump(self._build_metadata(), f, indent=2)
        return output_path

    def run(self, track: bool = True, promote: bool = False) -> dict:
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
            self._log_step("INGEST")
            df_raw = self.data_ingestor.fetch_data()

            self._log_step("CLEAN")
            df_clean = self.data_ingestor.clean(df_raw)

            self._log_step("SPLIT")
            df_selected = self._select_training_columns(df_clean)
            X = df_selected.drop(columns=[self.config.data.target_column])
            y = df_selected[self.config.data.target_column]
            self.X_train, self.X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.data.test_size,
                random_state=self.config.data.random_state,
            )

            self._log_step("FIT PREPROCESSOR")
            if track:
                mlflow.sklearn.autolog(disable=True)
            train_df = self.X_train.copy()
            train_df[self.config.data.target_column] = y_train
            train_df = self.data_ingestor.remove_outliers(train_df)
            self.X_train = train_df.drop(columns=[self.config.data.target_column])
            y_train = train_df[self.config.data.target_column]
            self.X_train_transformed = self.preprocessor.fit_transform(self.X_train)
            X_test_transformed = self.preprocessor.transform(self.X_test)

            self._log_step("FIT MODEL")
            if track:
                mlflow.sklearn.autolog(disable=False)
            self.model = TrainerFactory.get_model(self.config)
            self.model.fit(self.X_train_transformed, y_train.to_numpy())

            self._log_step("EVALUATE")
            test_pred = np.asarray(self.model.predict(X_test_transformed), dtype=float)
            self.metrics, self.prediction_interval = evaluate_predictions(y_test, test_pred)

            if track:
                self._log_step("LOG ARTIFACTS")
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    self.preprocessor.save(tmp_path / "preprocessor.pkl")
                    self.config_manager.save_config(str(tmp_path / "config.yaml"))
                    with open(tmp_path / "metadata.json", "w") as f:
                        json.dump(self._build_metadata(), f, indent=2)
                    mlflow.log_artifacts(str(tmp_path))

            should_promote = False
            if track and promote:
                self._log_step("GATE & REGISTER")
                run_id = mlflow.active_run().info.run_id
                should_promote = self.registry.evaluate_and_promote(
                    run_id=run_id,
                    current_metric=self.metrics["test"]["r2"],
                )
                if should_promote:
                    save_local_production_from_objects(
                        model=self.model,
                        preprocessor=self.preprocessor,
                        metadata=self._build_metadata(),
                        config_dict={
                            "data": self.config.data.model_dump(),
                            "preprocessing": self.config.preprocessing.model_dump(),
                            "model": self.config.model.model_dump(),
                            "training": self.config.training.model_dump(),
                        },
                    )

            self._log_step("DONE")
            self.metrics["promote_to_production"] = bool(should_promote)
            return self.metrics
