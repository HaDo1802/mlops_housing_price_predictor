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
from sklearn.model_selection import train_test_split

from predictor.config import ConfigManager, MLConfig
from predictor.data_ingest import DataIngestor
from predictor.models import TrainerFactory
from predictor.preprocessor import ProductionPreprocessor
from predictor.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from predictor.utils import evaluate_predictions

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrate ingestion, preprocessing, training, evaluation, and promotion."""

    def __init__(self, config_path: str = "conf/config.yaml"):
        self.config_path = Path(config_path)
        self.config_manager = ConfigManager(str(self.config_path))
        self.config: MLConfig = self.config_manager.config

        self.data_ingestor = DataIngestor(self.config)
        self.preprocessor = ProductionPreprocessor(
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            scaling_method=self.config.preprocessing.scaling_method,
            encoding_method=self.config.preprocessing.encoding_method,
            target_transform=self.config.preprocessing.target_transform,
            verbose=True,
        )

        self.model = None
        self.metrics = None
        self.prediction_interval = None

    @staticmethod
    def _log_step(step: str) -> None:
        logger.info("")
        logger.info("========== %s ==========", step)

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
            mlflow.set_experiment(self.config.training.experiment_name)

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
            df_clean = self.data_ingestor.remove_outliers(df_clean)

            self._log_step("SPLIT")
            df_selected = self.data_ingestor.select_training_columns(df_clean)
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
            self.X_train_transformed = self.preprocessor.fit_transform(self.X_train)
            X_test_transformed = self.preprocessor.transform(self.X_test)

            self._log_step("FIT MODEL")
            if track:
                mlflow.sklearn.autolog(log_models=False)
            self.model = TrainerFactory.get_model(self.config)
            self.model.fit(self.X_train_transformed, y_train.to_numpy())

            self._log_step("EVALUATE")
            test_pred = np.asarray(self.model.predict(X_test_transformed), dtype=float)
            self.metrics, self.prediction_interval = evaluate_predictions(y_test, test_pred)

            if track:
                self._log_step("LOG ARTIFACTS")
                mlflow.sklearn.log_model(self.model, artifact_path="model")
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    self.preprocessor.save(tmp_path / "preprocessor.pkl")
                    self.config_manager.save_config(str(tmp_path / "config.yaml"))
                    with open(tmp_path / "metadata.json", "w") as f:
                        json.dump(self._build_metadata(), f, indent=2)
                    mlflow.log_artifacts(str(tmp_path))

            self._log_step("DONE")
            self.metrics["promote_to_production"] = False
            return self.metrics
