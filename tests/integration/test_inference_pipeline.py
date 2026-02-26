from src.housing_predictor.pipelines.inference import InferencePipeline


def test_inference_pipeline_importable():
    assert InferencePipeline is not None
