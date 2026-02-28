import pandas as pd
import pytest


@pytest.fixture
def sample_housing_df():
    return pd.DataFrame(
        {
            "bedrooms": [2, 3, 3, 4, 5, 3],
            "bathrooms": [1.5, 2.0, 2.5, 3.0, 4.0, 2.0],
            "livingarea": [950, 1400, 1650, 2200, 3100, 1500],
            "latitude": [36.12, 36.15, 36.18, 36.08, 36.21, 36.10],
            "longitude": [-115.17, -115.23, -115.19, -115.30, -115.16, -115.20],
            "propertytype": [
                "CONDO",
                "SINGLE_FAMILY",
                "TOWNHOUSE",
                "SINGLE_FAMILY",
                "SINGLE_FAMILY",
                "CONDO",
            ],
            "vegas_district": [
                "Spring Valley",
                "Summerlin",
                "Downtown Las Vegas",
                "Centennial",
                "Summerlin",
                "Winchester",
            ],
            "price": [225000, 420000, 355000, 610000, 980000, 295000],
        }
    )
