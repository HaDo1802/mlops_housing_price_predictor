"""Training pipeline entrypoint."""

import logging

from src.housing_predictor.pipelines.training import TrainingPipeline


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    metrics = TrainingPipeline("conf/config.yaml").run()
    print(metrics)
