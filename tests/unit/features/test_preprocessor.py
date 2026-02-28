from housing_predictor.features.preprocessor import ProductionPreprocessor


def test_preprocessor_fit_transform(sample_housing_df):
    X = sample_housing_df.drop(columns=["price"])
    pre = ProductionPreprocessor(verbose=False)
    arr = pre.fit_transform(X)
    assert arr.shape[0] == len(X)
    assert pre.is_fitted is True
