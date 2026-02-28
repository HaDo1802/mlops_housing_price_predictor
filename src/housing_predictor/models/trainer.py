"""Model training utilities."""

import inspect

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge

MODEL_REGISTRY = {
    "gradient_boosting": GradientBoostingRegressor,
    "hist_gradient_boosting": HistGradientBoostingRegressor,
    "random_forest": RandomForestRegressor,
    "ridge": Ridge,
}


class ModelTrainer:
    """Thin model factory/trainer with optional log-target training."""

    def __init__(
        self,
        model_type: str,
        hyperparameters: dict,
        random_state: int = 42,
        use_log_target: bool = True,
    ):
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

        if "random_state" in valid_params:
            model_params["random_state"] = random_state
        else:
            model_params.pop("random_state", None)

        self.base_model = model_class(**model_params)
        self.use_log_target = use_log_target

        if use_log_target:
            # Train on log scale; predict back on original price scale.
            self.model = TransformedTargetRegressor(
                regressor=self.base_model,
                func=np.log1p,
                inverse_func=np.expm1,
                check_inverse=False,
            )
        else:
            self.model = self.base_model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def get_inner_model(self):
        """
        Return the underlying estimator, even when wrapped by
        TransformedTargetRegressor.
        """
        if isinstance(self.model, TransformedTargetRegressor):
            return getattr(self.model, "regressor_", self.base_model)
        return self.model
