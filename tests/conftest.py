import pandas as pd
import pytest


@pytest.fixture
def sample_housing_df():
    return pd.DataFrame(
        {
            "Overall Qual": [5, 6, 7, 8, 6, 5],
            "Year Built": [2000, 1990, 1980, 2010, 2005, 1995],
            "Neighborhood": ["NAmes", "CollgCr", "NAmes", "Edwards", "NAmes", "BrkSide"],
            "SalePrice": [150000, 180000, 170000, 230000, 210000, 160000],
        }
    )
