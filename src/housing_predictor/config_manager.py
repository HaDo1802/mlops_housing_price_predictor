"""Configuration management helpers for training and serving."""

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
    outlier_iqr_multiplier: float = 1.5
    outlier_ppsf_min: float = 80.0
    outlier_ppsf_max: float = 1000.0
    outlier_min_livingarea: float = 100.0
    scale_features: bool = True
    scaling_method: str = "standard"
    encode_categorical: bool = True
    encoding_method: str = "onehot"
    target_transform: str = "log1p"
    interval_num_segments: int = 5
    interval_min_segment_size: int = 10


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
    registry_model_name: str = "housing_price_predictor"
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

    @staticmethod
    def _normalize_features(raw_features: dict) -> dict:
        """Accept both {'numeric':...} and {'features': {'numeric':...}} shapes."""
        if not raw_features:
            return {}
        if "features" in raw_features and "numeric" not in raw_features:
            return raw_features.get("features") or {}
        return raw_features

    def _load_layered_config(self) -> dict:
        root = self.config_path.parent
        base_dir = root / "base"
        env_name = self.env or "local"

        # Step 1: load base config
        cfg = {
            "data": self._load_yaml(base_dir / "data.yaml"),
            "features": self._normalize_features(
                self._load_yaml(base_dir / "features.yaml")
            ),
            "preprocessing": self._load_yaml(base_dir / "preprocessing.yaml"),
            "model": self._load_yaml(base_dir / "model.yaml"),
            "training": self._load_yaml(base_dir / "training.yaml"),
        }
        # Step 2: override with environment-specific values
        env_data = self._load_yaml(root / env_name / "data.yaml")
        cfg["data"].update(env_data)
        # Step 3: override with whatever is in config.yaml  ← THIS WAS MISSING
        main_config = self._load_yaml(root / "config.yaml")

        if "features" in main_config:
            main_config["features"] = self._normalize_features(main_config["features"])

        # Backward compatibility for flat model keys in conf/config.yaml
        if "model" not in main_config:
            flat_model = {}
            if "model_type" in main_config:
                flat_model["model_type"] = main_config["model_type"]
            if "hyperparameters" in main_config:
                flat_model["hyperparameters"] = main_config["hyperparameters"]
            if "random_state" in main_config:
                flat_model["random_state"] = main_config["random_state"]
            if flat_model:
                main_config["model"] = flat_model

        for section in ["data", "features", "preprocessing", "model", "training"]:
            if section in main_config and main_config[section]:
                cfg[section].update(main_config[section])
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
            features=FeatureSelectionConfig(
                **self._normalize_features(config_dict.get("features", {}))
            ),
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
