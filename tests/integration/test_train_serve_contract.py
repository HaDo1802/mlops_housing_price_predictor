import json
import pickle

import numpy as np
import yaml

from housing_predictor.pipelines.inference import InferencePipeline
from housing_predictor.pipelines.training import TrainingPipeline


def test_training_artifacts_can_be_loaded_by_inference_pipeline(tmp_path, raw_training_df):
    """
    End-to-end contract test:
    train on a tiny local dataset, save artifacts, then load them through the
    production inference pipeline and score one row.
    """
    data_path = tmp_path / "train.csv"
    raw_training_df.to_csv(data_path, index=False)

    with open("conf/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["data"]["raw_data_path"] = str(data_path)
    config["preprocessing"]["handle_outliers"] = False
    config["model"]["model_type"] = "ridge"
    config["model"]["hyperparameters"] = {}

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    training = TrainingPipeline(str(config_path))
    training.run(track=False, promote=False)

    artifact_dir = training.save_artifacts(str(tmp_path / "artifacts"))
    inference = InferencePipeline(local_model_dir=str(artifact_dir))

    row = training.X_test.head(1).copy()
    preds = inference.predict(row)

    assert len(preds) == 1
    assert np.isfinite(preds[0])
    assert inference.preprocessor.numeric_features == training.preprocessor.numeric_features
    assert inference.preprocessor.categorical_features == training.preprocessor.categorical_features

    with open(artifact_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    with open(artifact_dir / "model.pkl", "rb") as f:
        assert pickle.load(f) is not None

    assert metadata["feature_names"] == training.preprocessor.get_feature_names()
