"""Hyperparameter tuning entrypoint."""

import logging
from itertools import product
from pathlib import Path

import mlflow
import yaml

from src.housing_predictor.pipelines.training import TrainingPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_hyperparameter_search() -> None:
    mlflow.set_experiment("house_price_hyperparameter_search")

    learning_rates = [0.05, 0.1, 0.15]
    max_depths = [3, 5, 7]
    n_estimators_list = [50, 100]

    config_path = "conf/config.yaml"
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    total_runs = len(learning_rates) * len(max_depths) * len(n_estimators_list)
    logger.info("Starting hyperparameter search: %s total runs", total_runs)

    results = []
    run_num = 0

    for lr, depth, n_est in product(learning_rates, max_depths, n_estimators_list):
        run_num += 1
        config = base_config.copy()
        config["model"]["hyperparameters"]["learning_rate"] = lr
        config["model"]["hyperparameters"]["max_depth"] = depth
        config["model"]["hyperparameters"]["n_estimators"] = n_est
        config["training"]["run_name"] = f"lr{lr}_depth{depth}_nest{n_est}"

        temp_config_path = "conf/temp_config.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f)

        try:
            metrics = TrainingPipeline(temp_config_path).run()
            results.append(
                {
                    "learning_rate": lr,
                    "max_depth": depth,
                    "n_estimators": n_est,
                    "test_r2": metrics["test"]["r2"],
                    "test_rmse": metrics["test"]["rmse"],
                }
            )
            logger.info("Run %s/%s complete", run_num, total_runs)
        except Exception as exc:
            logger.error("Run %s failed: %s", run_num, exc)

    Path("conf/temp_config.yaml").unlink(missing_ok=True)

    if not results:
        logger.warning("No successful tuning runs")
        return

    best = sorted(results, key=lambda x: x["test_r2"], reverse=True)[0]
    logger.info("Best config: %s", best)


if __name__ == "__main__":
    run_hyperparameter_search()
