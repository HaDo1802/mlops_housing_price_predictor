"""Shared feature schema and API field mappings."""

MODEL_FEATURES = [
    "bedrooms",
    "bathrooms",
    "livingarea",
    "latitude",
    "longitude",
    "propertytype",
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
]

FEATURE_DISPLAY_LABELS = {
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "livingarea": "Living Area (sqft)",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "propertytype": "Property Type",
}

CATEGORICAL_OPTIONS = {
    "propertytype": [
        "SINGLE_FAMILY",
        "TOWNHOUSE",
        "CONDO",
        "MULTI_FAMILY",
        "MOBILE",
    ],
}

TARGET_COLUMN = "price"
