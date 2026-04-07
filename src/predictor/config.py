"""Configuration models and helpers."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    target_column: str
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42


class PreprocessingConfig(BaseModel):
    handle_missing: str = "mean"
    handle_outliers: bool = True
    min_price: float = 100000.0
    max_price: float = 5000000.0
    min_living_area: float = 500.0
    exclude_property_types: list[str] = Field(default_factory=list)
    scale_features: bool = True
    scaling_method: str = "standard"
    encode_categorical: bool = True
    encoding_method: str = "onehot"
    target_transform: str = "log1p"


class ModelConfig(BaseModel):
    model_type: str = "hist_gradient_boosting"
    random_state: int = 42
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    cv_folds: int = 5
    optimize_metric: str = "r2"


class TrainingConfig(BaseModel):
    experiment_name: str = "house_price_prediction"
    run_name: str = "run_001"
    registry_model_name: str = "housing_price_predictor"
    log_metrics: bool = True
    save_artifacts: bool = True
    track_experiments: bool = True
    model_output_dir: str = "models"
    log_dir: str = "logs"


class MLConfig(BaseModel):
    data: DataConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    training: TrainingConfig


class ConfigManager:
    def __init__(self, config_path: str = "conf/config.yaml"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        self.config = MLConfig(
            data=DataConfig(**config_dict["data"]),
            preprocessing=PreprocessingConfig(**config_dict["preprocessing"]),
            model=ModelConfig(**config_dict["model"]),
            training=TrainingConfig(**config_dict["training"]),
        )

    def save_config(self, output_path: str) -> None:
        with open(output_path, "w") as f:
            yaml.safe_dump(self.config.model_dump(), f, sort_keys=False)
