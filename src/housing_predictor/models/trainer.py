"""Model training utilities."""

# src/housing_predictor/models/trainer.py

import inspect

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

MODEL_REGISTRY = {
    "gradient_boosting": GradientBoostingRegressor,
    "random_forest": RandomForestRegressor,
    "ridge": Ridge,
}

class ModelTrainer:
    def __init__(self, model_type: str, hyperparameters: dict, random_state: int = 42):
        if model_type not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_type: '{model_type}'. "
                f"Choose from: {list(MODEL_REGISTRY.keys())}"
            )

        model_class = MODEL_REGISTRY[model_type]
        model_params = dict(hyperparameters or {})

        valid_params = set(inspect.signature(model_class.__init__).parameters.keys())
        valid_params.discard("self")
        invalid_params = sorted(set(model_params.keys()) - valid_params)
        if invalid_params:
            raise ValueError(
                f"Invalid hyperparameters for '{model_type}': {invalid_params}. "
                f"Allowed keys include: {sorted(valid_params)}"
            )

        # Prefer explicit pipeline random_state and avoid duplicate kwargs.
        if "random_state" in valid_params:
            model_params["random_state"] = random_state
        else:
            model_params.pop("random_state", None)

        self.model = model_class(**model_params)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model
