"""Shared feature schema and API field mappings."""

MODEL_FEATURES = [
    "bedrooms",
    "bathrooms",
    "livingarea",
    "latitude",
    "longitude",
    "propertytype",
    "vegas_district",
]

API_TO_MODEL_FIELDS = {
    "bedrooms": "bedrooms",
    "bathrooms": "bathrooms",
    "livingarea": "livingarea",
    "living_area": "livingarea",
    "latitude": "latitude",
    "longitude": "longitude",
    "propertytype": "propertytype",
    "property_type": "propertytype",
    "vegas_district": "vegas_district",
}

NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "livingarea",
    "latitude",
    "longitude",
]

CATEGORICAL_FEATURES = [
    "propertytype",
    "vegas_district",
]

TARGET_COLUMN = "price"
