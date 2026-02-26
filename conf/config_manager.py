"""Configuration Management Module."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class DataConfig:
    raw_data_path: str
    target_column: str
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42


@dataclass
class PreprocessingConfig:
    handle_missing: str = "mean"
    handle_outliers: bool = True
    outlier_method: str = "iqr"
    scale_features: bool = True
    scaling_method: str = "standard"
    encode_categorical: bool = True
    encoding_method: str = "onehot"


@dataclass
class ModelConfig:
    model_type: str = "random_forest"
    random_state: int = 42
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    cv_folds: int = 5
    optimize_metric: str = "r2"


@dataclass
class TrainingConfig:
    experiment_name: str = "house_price_prediction"
    run_name: str = "run_001"
    log_metrics: bool = True
    save_artifacts: bool = True
    track_experiments: bool = True
    model_output_dir: str = "models"
    log_dir: str = "logs"


@dataclass
class FeatureSelectionConfig:
    numeric: list = field(default_factory=list)
    categorical: list = field(default_factory=list)


@dataclass
class MLConfig:
    data: DataConfig
    features: FeatureSelectionConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    training: TrainingConfig


class ConfigManager:
    """Loads single-file config or layered conf/base + env overrides."""

    def __init__(self, config_path: str = "conf/config.yaml", env: str | None = None):
        self.config_path = Path(config_path)
        self.env = env
        self.config = self._load_config()

    def _load_yaml(self, path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def _load_layered_config(self) -> dict:
        root = self.config_path.parent
        base_dir = root / "base"
        env_name = self.env or "local"

        cfg = {
            "data": self._load_yaml(base_dir / "data.yaml"),
            "features": self._load_yaml(root / "config.yaml").get("features", {}),
            "preprocessing": self._load_yaml(base_dir / "preprocessing.yaml"),
            "model": self._load_yaml(base_dir / "model.yaml"),
            "training": self._load_yaml(base_dir / "training.yaml"),
        }

        env_data = self._load_yaml(root / env_name / "data.yaml")
        cfg["data"].update(env_data)
        return cfg

    def _load_config(self) -> MLConfig:
        if self.config_path.is_dir() or (self.config_path.parent / "base").exists():
            config_dict = self._load_layered_config()
        else:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
            config_dict = self._load_yaml(self.config_path)

        return MLConfig(
            data=DataConfig(**config_dict["data"]),
            features=FeatureSelectionConfig(**config_dict.get("features", {})),
            preprocessing=PreprocessingConfig(**config_dict["preprocessing"]),
            model=ModelConfig(**config_dict["model"]),
            training=TrainingConfig(**config_dict["training"]),
        )

    def get_config(self) -> MLConfig:
        return self.config

    def save_config(self, output_path: str) -> None:
        config_dict = {
            "data": self.config.data.__dict__,
            "features": self.config.features.__dict__,
            "preprocessing": self.config.preprocessing.__dict__,
            "model": self.config.model.__dict__,
            "training": self.config.training.__dict__,
        }
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
