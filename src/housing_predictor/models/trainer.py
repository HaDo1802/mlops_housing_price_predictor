"""Model training utilities."""

from sklearn.ensemble import GradientBoostingRegressor


class GradientBoostingTrainer:
    """Thin wrapper around GradientBoostingRegressor."""

    def __init__(self, hyperparameters: dict, random_state: int = 42):
        self.model = GradientBoostingRegressor(
            **hyperparameters,
            random_state=random_state,
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model
