"""
Housing Price Prediction - Streamlit Web Application

This app provides a user-friendly interface for predicting housing prices
using the trained Gradient Boosting model.

Features:
- Interactive input forms for all required features
- Input validation
- Prediction with uncertainty intervals
- Feature importance visualization
- Professional result formatting

Usage:
    streamlit run serving/app/streamlit_app.py
"""

import logging
import os
import sys
import uuid
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from predictor.schema import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    OPTIONAL_FEATURE_DEFAULTS,
)
from predictor.predict import InferencePipeline
from serving.api.feature_map import (
    API_TO_MODEL_FIELDS,
    CATEGORICAL_OPTIONS,
    FEATURE_DISPLAY_LABELS,
    VEGAS_DISTRICT_CENTROIDS,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")


# Page configuration
st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
    }
    .interval-text {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-top: 0.5rem;
    }
    .feature-section {
        background-color: #f9f9f9;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ff4444;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e6f7ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1890ff;
        margin: 1rem 0;
    }
    </style>
""",
    unsafe_allow_html=True,
)


NUMERIC_INPUT_CONFIG = {
    "bedrooms": {"min_value": 0, "max_value": 20, "value": 3, "step": 1},
    "bathrooms": {"min_value": 0.0, "max_value": 20.0, "value": 2.0, "step": 0.5},
    "livingarea": {
        "min_value": 100.0,
        "max_value": 20000.0,
        "value": 1800.0,
        "step": 50.0,
    },
    "latitude": {"min_value": 35.5, "max_value": 36.5, "value": 36.1, "step": 0.0001},
    "longitude": {
        "min_value": -115.5,
        "max_value": -114.5,
        "value": -115.17,
        "step": 0.0001,
    },
}


@st.cache_resource
def load_inference_pipeline():
    """Load the production inference pipeline once per Streamlit process."""
    return InferencePipeline(model_name="housing_price_predictor", stage="Production")


def get_model_info() -> dict | None:
    """Return model metadata from the loaded inference pipeline."""
    try:
        return load_inference_pipeline().get_model_info()
    except Exception as exc:
        logger.error("Model load failed: %s", exc)
        return None


def resolve_feature_spec() -> tuple[list[str], list[str], dict[str, str], dict[str, list[str]], list[str]]:
    """Return the local raw feature contract used by the model and UI."""
    return (
        list(NUMERIC_FEATURES),
        list(CATEGORICAL_FEATURES),
        dict(FEATURE_DISPLAY_LABELS),
        dict(CATEGORICAL_OPTIONS),
        list(OPTIONAL_FEATURE_DEFAULTS),
    )


def _fill_missing_location_from_district(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill latitude/longitude from vegas_district centroids when needed."""
    df = df.copy()
    if "vegas_district" not in df.columns:
        return df

    if "latitude" not in df.columns:
        df["latitude"] = np.nan
    if "longitude" not in df.columns:
        df["longitude"] = np.nan

    for district, centroid in VEGAS_DISTRICT_CENTROIDS.items():
        mask = (
            df["vegas_district"].eq(district)
            & (df["latitude"].isna() | df["longitude"].isna())
        )
        if mask.any():
            df.loc[mask, "latitude"] = df.loc[mask, "latitude"].fillna(
                float(centroid["latitude"])
            )
            df.loc[mask, "longitude"] = df.loc[mask, "longitude"].fillna(
                float(centroid["longitude"])
            )
    return df


def build_model_row(inputs: dict) -> dict:
    """Convert UI inputs into the raw model feature row."""
    row = {k: v for k, v in inputs.items() if v is not None}
    if (
        ("latitude" not in row or "longitude" not in row)
        and "vegas_district" in row
        and row["vegas_district"] in VEGAS_DISTRICT_CENTROIDS
    ):
        centroid = VEGAS_DISTRICT_CENTROIDS[row["vegas_district"]]
        row.setdefault("latitude", float(centroid["latitude"]))
        row.setdefault("longitude", float(centroid["longitude"]))
    return row


def _normalize_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize file-upload columns into the raw model feature contract."""
    rename_map = {
        column: API_TO_MODEL_FIELDS[column]
        for column in df.columns
        if column in API_TO_MODEL_FIELDS
    }
    normalized = df.rename(columns=rename_map).copy()
    normalized = _fill_missing_location_from_district(normalized)
    return normalized


def request_prediction(row: dict) -> dict:
    """Run a single prediction directly through the local inference pipeline."""
    pipeline = load_inference_pipeline()
    df = pd.DataFrame([row])
    preds, lower_bounds, upper_bounds = pipeline.predict_with_uncertainty(df)
    pred = float(preds[0])
    try:
        top_features = pipeline.get_feature_importance(top_n=5).to_dict("records")
    except Exception:
        top_features = None
    return {
        "prediction": pred,
        "confidence_interval": {
            "lower": float(pred + lower_bounds[0]),
            "upper": float(pred + upper_bounds[0]),
        },
        "top_features": top_features,
    }


def request_file_prediction(file_name: str, file_bytes: bytes, mime_type: str) -> bytes:
    """Run local batch prediction and return CSV bytes."""
    if file_name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    elif file_name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        raise RuntimeError("Unsupported file type")

    pipeline = load_inference_pipeline()
    normalized = _normalize_input_dataframe(df)
    preds, lower_bounds, upper_bounds = pipeline.predict_with_uncertainty(normalized)

    out = df.copy()
    out["prediction"] = preds
    out["lower_bound"] = preds + lower_bounds
    out["upper_bound"] = preds + upper_bounds
    return out.to_csv(index=False).encode("utf-8")


def validate_inputs(inputs: dict, required_features: list) -> tuple:
    """
    Validate user inputs.

    Returns:
        (is_valid: bool, missing_fields: list, error_messages: list)
    """
    missing_fields = []
    error_messages = []

    # Check for missing fields
    for feature in required_features:
        if feature not in inputs or inputs[feature] is None or inputs[feature] == "":
            missing_fields.append(feature)

    # Check numeric ranges using per-feature input config.
    for feature, value in inputs.items():
        if feature in required_features:
            # Skip if already missing
            if feature in missing_fields:
                continue

            # Validate numeric ranges
            if isinstance(value, (int, float)):
                cfg = NUMERIC_INPUT_CONFIG.get(feature, {})
                min_value = cfg.get("min_value")
                max_value = cfg.get("max_value")
                if min_value is not None and value < min_value:
                    error_messages.append(f"{feature}: Must be >= {min_value}")
                if max_value is not None and value > max_value:
                    error_messages.append(f"{feature}: Must be <= {max_value}")

    is_valid = len(missing_fields) == 0 and len(error_messages) == 0

    return is_valid, missing_fields, error_messages


def create_input_form(
    numeric_features: list[str],
    categorical_features: list[str],
    display_labels: dict[str, str],
    categorical_options: dict[str, list[str]],
) -> tuple[dict, list[str]]:
    """Create input form from resolved feature specification."""

    st.markdown(
        '<div class="main-header">🏠 Housing Price Predictor</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Enter property details to predict the sale price</div>',
        unsafe_allow_html=True,
    )

    # Initialize session state for inputs
    if "inputs" not in st.session_state:
        st.session_state.inputs = {}

    inputs = {}
    st.subheader("Property Inputs")
    all_features = list(numeric_features) + list(categorical_features)
    if not all_features:
        st.info("No input features available from API/model metadata.")
        return inputs, all_features

    col_left, col_right = st.columns(2)
    for idx, feature in enumerate(all_features):
        col = col_left if idx % 2 == 0 else col_right
        with col:
            label = display_labels.get(feature, feature.replace("_", " ").title())
            if feature in numeric_features:
                params = NUMERIC_INPUT_CONFIG.get(
                    feature, {"min_value": 0.0, "value": 0.0, "step": 1.0}
                )
                inputs[feature] = st.number_input(
                    label,
                    key=f"input_{feature}",
                    help=f"Enter {label.lower()}",
                    **params,
                )
            else:
                options = categorical_options.get(feature)
                if options:
                    val = st.selectbox(
                        label,
                        options=[""] + options,
                        help=f"Select {label.lower()}",
                        key=f"input_{feature}",
                    )
                    inputs[feature] = None if val == "" else val
                else:
                    text = st.text_input(
                        label,
                        value="",
                        help=f"Enter {label.lower()}",
                        key=f"input_{feature}",
                    )
                    inputs[feature] = None if text.strip() == "" else text.strip()

    return inputs, numeric_features + categorical_features


def display_prediction_results(
    prediction, lower, upper, top_features, confidence_level: float = 95.0
):
    """Display prediction results in a professional format"""

    st.markdown("---")
    st.markdown("## 🎯 Prediction Results")

    # Main prediction box
    st.markdown(
        f"""
    <div class="prediction-box">
        <div class="prediction-value">${prediction:,.0f}</div>
        <div class="interval-text">
            Confidence Interval: ${lower:,.0f} - ${upper:,.0f}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Create two columns for additional info
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Prediction Statistics")

        # Calculate margin of error
        margin = (upper - lower) / 2
        margin_pct = (margin / prediction) * 100

        stats_df = pd.DataFrame(
            {
                "Metric": [
                    "Predicted Price",
                    "Lower Bound",
                    "Upper Bound",
                    "Margin of Error",
                    "Confidence Level",
                ],
                "Value": [
                    f"${prediction:,.0f}",
                    f"${lower:,.0f}",
                    f"${upper:,.0f}",
                    f"${margin:,.0f} (±{margin_pct:.1f}%)",
                    f"{confidence_level:.1f}%",
                ],
            }
        )

        st.dataframe(stats_df, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("### 🔑 Top 5 Most Important Features")

        if top_features is None or top_features.empty:
            st.info("Feature importance is not available from the API.")
        else:
            # Create horizontal bar chart for top features
            fig = go.Figure(
                go.Bar(
                    x=top_features["importance"].values,
                    y=top_features["feature"].values,
                    orientation="h",
                    marker=dict(
                        color=top_features["importance"].values,
                        colorscale="Blues",
                        showscale=False,
                    ),
                    text=top_features["importance"].apply(lambda x: f"{x:.4f}"),
                    textposition="auto",
                )
            )

            fig.update_layout(
                title="Feature Importance Scores",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis={"categoryorder": "total ascending"},
            )

            st.plotly_chart(fig, use_container_width=True)

    # Price range visualization
    st.markdown("### 📈 Price Range Visualization")

    fig = go.Figure()

    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=["Lower Bound", "Prediction", "Upper Bound"],
            y=[lower, prediction, upper],
            mode="markers+lines",
            marker=dict(size=[15, 25, 15], color=["#ff7f0e", "#1f77b4", "#ff7f0e"]),
            line=dict(color="#1f77b4", width=2),
            name="Price Range",
        )
    )

    # Add shaded area for confidence interval
    fig.add_trace(
        go.Scatter(
            x=["Lower Bound", "Prediction", "Upper Bound"],
            y=[lower, lower, lower],
            fill=None,
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=["Lower Bound", "Prediction", "Upper Bound"],
            y=[upper, upper, upper],
            fill="tonexty",
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            fillcolor="rgba(31, 119, 180, 0.2)",
            name="95% Confidence Interval",
        )
    )

    fig.update_layout(
        title="Predicted Price with Confidence Interval",
        xaxis_title="",
        yaxis_title="Price ($)",
        height=400,
        hovermode="x unified",
        yaxis=dict(tickformat="$,.0f"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add interpretation
    st.markdown(
        f"""
    <div class="info-box">
        <strong>💡 How to interpret these results:</strong><br>
        • The predicted sale price is <strong>${prediction:,.0f}</strong><br>
        • We are {confidence_level:.1f}% confident the actual price will be between <strong>${lower:,.0f}</strong> and <strong>${upper:,.0f}</strong><br>
        • The top 5 features shown above had the most influence on this prediction<br>
        • The margin of error is <strong>±${margin:,.0f}</strong> ({margin_pct:.1f}%)
    </div>
    """,
        unsafe_allow_html=True,
    )


def main():
    """Main application logic"""

    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1946/1946488.png", width=100)
        st.title("About")
        st.markdown(
            """
        This application predicts property prices using the deployed
        **Housing Predictor** model loaded directly from the production artifact snapshot.
        
        **Core input features:**
        - Bedrooms, Bathrooms, Living Area
        - Latitude, Longitude
        - Property Type
        
        **Instructions:**
        1. Fill in the required property details
        2. Click "Predict Price" button
        3. Review prediction with confidence interval
        4. Check which features influenced the prediction most
        """
        )

        st.markdown("**Input Requirements:**")
        st.markdown(
            f"""
        - `bedrooms`: {NUMERIC_INPUT_CONFIG['bedrooms']['min_value']} to {NUMERIC_INPUT_CONFIG['bedrooms']['max_value']}
        - `bathrooms`: {NUMERIC_INPUT_CONFIG['bathrooms']['min_value']} to {NUMERIC_INPUT_CONFIG['bathrooms']['max_value']}
        - `livingarea`: {NUMERIC_INPUT_CONFIG['livingarea']['min_value']} to {NUMERIC_INPUT_CONFIG['livingarea']['max_value']} sqft
        - `latitude` / `longitude`: optional if `vegas_district` is provided
        - `normalized_lot_area_value`: optional, defaults to {OPTIONAL_FEATURE_DEFAULTS['normalized_lot_area_value']:.0f} sqft when omitted
        - `propertytype`: select one of the provided categories
        """
        )

        st.markdown("---")
        st.markdown("**Model Info:**")
        model_info = get_model_info()
        if model_info:
            metadata = model_info.get("metadata", {})
            st.info(f"Model: {metadata.get('model_type', 'unknown')}")
            st.info(f"Loaded from: {model_info.get('loaded_from', 'unknown')}")
            bucket = os.getenv("ARTIFACT_BUCKET")
            if bucket:
                st.caption(f"S3 bucket: `{bucket}`")
            interval_cfg = metadata.get("prediction_interval") or {}
            st.session_state.confidence_level = float(
                interval_cfg.get("coverage", 0.95) * 100
            )
        else:
            st.error("Model could not be loaded. Check S3 credentials or local production artifacts.")
            st.session_state.confidence_level = 95.0

        st.markdown("---")
        st.markdown("**Need Help?**")
        st.markdown(
            """
        - Bedrooms, bathrooms, living area, property type, and location are required
        - `normalized_lot_area_value` is optional and falls back to a default lot size
        - Numerical values must be within the allowed ranges shown in the form
        - Latitude/longitude should be in the target market bounds
        - Use exact categorical values shown in dropdowns
        """
        )

    # Main content
    (
        numeric_features,
        categorical_features,
        display_labels,
        categorical_options,
        optional_features,
    ) = resolve_feature_spec()
    inputs, all_features = create_input_form(
        numeric_features,
        categorical_features,
        display_labels,
        categorical_options,
    )
    required_features = [f for f in all_features if f not in set(optional_features)]

    st.markdown("---")
    st.markdown("## 📂 Batch Predictions (CSV/Excel)")
    st.markdown(
        "Upload a CSV or Excel file with the same feature columns to get predictions for multiple rows."
    )

    uploaded_file = st.file_uploader(
        "Upload file",
        type=["csv", "xlsx", "xls"],
        help="Accepted formats: CSV, XLSX, XLS",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        mime_type = uploaded_file.type or "application/octet-stream"

        st.info(f"Selected file: {file_name}")
        if st.button("🚀 Run Batch Prediction", type="primary"):
            with st.spinner("🔄 Processing file..."):
                try:
                    output_csv = request_file_prediction(
                        file_name=file_name,
                        file_bytes=file_bytes,
                        mime_type=mime_type,
                    )
                    output_name = f"{Path(file_name).stem}_predictions.csv"
                    st.success("✅ Batch predictions completed!")
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=output_csv,
                        file_name=output_name,
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"❌ Batch prediction failed: {str(e)}")
                    logger.error("Batch prediction error: %s", e, exc_info=True)

    # Predict button
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🔮 Predict Price", use_container_width=True, type="primary"
        )

    if predict_button:
        # Validate inputs
        is_valid, missing_fields, error_messages = validate_inputs(
            inputs, required_features
        )

        if not is_valid:
            # Display errors
            if missing_fields:
                st.markdown(
                    f"""
                <div class="error-message">
                    <strong>❌ Missing Required Fields:</strong><br>
                    {', '.join(missing_fields)}
                </div>
                """,
                    unsafe_allow_html=True,
                )

            if error_messages:
                st.markdown(
                    f"""
                <div class="error-message">
                    <strong>❌ Invalid Values:</strong><br>
                    {'<br>'.join(error_messages)}
                </div>
                """,
                    unsafe_allow_html=True,
                )

            st.warning(
                "⚠️ Please fill in all required fields with valid values before making a prediction."
            )

        else:
            # Make prediction
            with st.spinner("🔄 Calculating prediction..."):
                try:
                    row = build_model_row(inputs)
                    response = request_prediction(row)

                    prediction = response["prediction"]
                    lower = response["confidence_interval"]["lower"]
                    upper = response["confidence_interval"]["upper"]
                    top_features = response.get("top_features")
                    if top_features is not None:
                        top_features = pd.DataFrame(top_features)

                    st.session_state.last_result = {
                        "prediction_id": str(uuid.uuid4()),
                        "prediction": float(prediction),
                        "lower": float(lower),
                        "upper": float(upper),
                        "top_features": top_features,
                    }
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    logger.error(f"Prediction error: {str(e)}", exc_info=True)

    if "last_result" in st.session_state:
        result = st.session_state.last_result

        # Display results
        display_prediction_results(
            result["prediction"],
            result["lower"],
            result["upper"],
            result["top_features"],
            st.session_state.get("confidence_level", 95.0),
        )

        # Success message
        st.success("✅ Prediction completed successfully!")

        # Download button for results
        results_data = {
            "Predicted Price": [f"${result['prediction']:,.0f}"],
            "Lower Bound (95% CI)": [f"${result['lower']:,.0f}"],
            "Upper Bound (95% CI)": [f"${result['upper']:,.0f}"],
            "Margin of Error": [f"${(result['upper'] - result['lower']) / 2:,.0f}"],
        }
        results_df = pd.DataFrame(results_data)

        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Prediction Results",
            data=csv,
            file_name="house_price_prediction.csv",
            mime="text/csv",
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <small>Housing Price Predictor v1.0 | Built with Streamlit | 
        Powered by Gradient Boosting ML Model</small>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
