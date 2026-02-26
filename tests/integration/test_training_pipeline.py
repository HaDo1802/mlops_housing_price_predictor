from src.housing_predictor.pipelines.training import TrainingPipeline


def test_training_pipeline_importable():
    assert TrainingPipeline is not None
