from housing_predictor.data.splitter import DataSplitter


def test_splitter_sizes(sample_housing_df):
    splitter = DataSplitter(test_size=0.2, val_size=0.2, random_state=42, verbose=False)
    X_train, X_test, X_val, y_train, y_test, y_val = splitter.split_dataframe(
        sample_housing_df, target_col="price"
    )
    assert len(X_train) + len(X_test) + len(X_val) == len(sample_housing_df)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert len(y_val) == len(X_val)
