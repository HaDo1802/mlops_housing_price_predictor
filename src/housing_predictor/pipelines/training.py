"""Training pipeline orchestration layer."""

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn

from housing_predictor.config_manager import ConfigManager
from housing_predictor.data.loader import load_dataframe
from housing_predictor.data.splitter import DataSplitter
from housing_predictor.features.preprocessor import ProductionPreprocessor
from housing_predictor.models.evaluator import regression_metrics
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
            verbose=True,
        )
        self.trainer = ModelTrainer(
            model_type=self.config.model.model_type,  # now actually used
            hyperparameters=self.config.model.hyperparameters,
            random_state=self.config.model.random_state,
        )
        self.registry = ModelRegistryManager(
            registry_model_name=self.config.training.registry_model_name
        )

        self.metrics = None

    def load_and_select(self):
        df_raw = load_dataframe(self.config.data.raw_data_path)
        numeric = self.config.features.numeric
        categorical = self.config.features.categorical

        selected = numeric + categorical
        target_col = self.config.data.target_column
        if target_col not in selected:
            selected.append(target_col)

        available = [f for f in selected if f in df_raw.columns]
        self.df_selected = df_raw[available].copy()

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

    def train_and_eval(self):
        self.model = self.trainer.fit(self.X_train_transformed, self.y_train)
        test_pred = self.model.predict(self.X_test_transformed)
        val_pred = self.model.predict(self.X_val_transformed)
        self.metrics = {
            "test": regression_metrics(self.y_test, test_pred),
            "validation": regression_metrics(self.y_val, val_pred),
        }

    def _build_metadata(self) -> dict:
        return {
            "model_type": type(self.trainer.model).__name__,
            "model_key": self.config.model.model_type,
            "hyperparameters": self.config.model.hyperparameters,
            "test_metrics": self.metrics["test"],
            "val_metrics": self.metrics["validation"],
            "feature_names": self.preprocessor.get_feature_names(),
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
        mlflow.set_experiment(self.config.training.experiment_name)

        with mlflow.start_run(run_name=self.config.training.run_name):
            mlflow.log_params(
                {
                    "test_size": self.config.data.test_size,
                    "val_size": self.config.data.val_size,
                    "random_state": self.config.data.random_state,
                    "scaling_method": self.config.preprocessing.scaling_method,
                    **self.config.model.hyperparameters,
                }
            )
            mlflow.set_tag("model_type", type(self.trainer.model).__name__)
            mlflow.set_tag("model_key", self.config.model.model_type)
            mlflow.set_tag("project", "house_price_prediction")
            mlflow.set_tag("training_date", datetime.now(timezone.utc).isoformat())
            mlflow.set_tag("git_commit", self.registry.get_git_commit())

            self.load_and_select()
            self.split_data()
            self.preprocess_data()
            self.train_and_eval()

            mlflow.log_metrics(
                {
                    "test_r2": self.metrics["test"]["r2"],
                    "test_rmse": self.metrics["test"]["rmse"],
                    "test_mae": self.metrics["test"]["mae"],
                    "val_r2": self.metrics["validation"]["r2"],
                    "val_rmse": self.metrics["validation"]["rmse"],
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

            run_id = mlflow.active_run().info.run_id
            model_version = self.registry.register_model(run_id)
            self.registry.auto_promote_if_better(
                new_model_version=model_version,
                new_test_r2=self.metrics["test"]["r2"],
            )

            backup_path = self.save_artifacts()
            logger.info("Local experiment backup path: %s", backup_path)
            return self.metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = TrainingPipeline("conf/config.yaml")
    metrics = pipeline.run()
    print(metrics)
