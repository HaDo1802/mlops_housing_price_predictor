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

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from conf.config_manager import ConfigManager
from src.housing_predictor.monitoring.feedback_collector import save_feedback_record

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


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


@st.cache_resource
def load_config():
    """Load configuration (cached)"""
    try:
        config_manager = ConfigManager("conf/config.yaml")
        return config_manager.get_config()
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        st.stop()

@st.cache_data(ttl=60)
def fetch_model_info():
    """Fetch model metadata from the FastAPI service"""
    url = f"{API_BASE_URL}/model/info"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        logger.warning(
            "Model info request failed: %s - %s", response.status_code, response.text
        )
    except requests.RequestException as exc:
        logger.warning("Model info request error: %s", exc)
    return None


def build_api_payload(inputs: dict) -> dict:
    """Map Streamlit inputs to FastAPI payload keys"""
    feature_map = {
        "Lot Area": "lot_area",
        "Total Bsmt SF": "total_bsmt_sf",
        "1st Flr SF": "1st Flr SF",
        "2nd Flr SF": "2nd Flr SF",
        "Gr Liv Area": "gr_liv_area",
        "Garage Area": "garage_area",
        "Overall Qual": "overall_qual",
        "Overall Cond": "overall_cond",
        "Year Built": "year_built",
        "Year Remod/Add": "year_remod_add",
        "Bedroom AbvGr": "bedroom_abvgr",
        "Full Bath": "full_bath",
        "Half Bath": "half_bath",
        "TotRms AbvGrd": "totrms_abvgrd",
        "Fireplaces": "fireplaces",
        "Garage Cars": "garage_cars",
        "Neighborhood": "neighborhood",
        "MS Zoning": "ms_zoning",
        "Bldg Type": "bldg_type",
        "House Style": "house_style",
        "Foundation": "foundation",
        "Central Air": "central_air",
        "Garage Type": "garage_type",
    }

    payload = {}
    for feature, value in inputs.items():
        api_key = feature_map.get(feature)
        if api_key is not None:
            payload[api_key] = value
    return payload


def request_prediction(payload: dict) -> dict:
    """Call FastAPI prediction endpoint"""
    url = f"{API_BASE_URL}/predict"
    try:
        response = requests.post(url, json=payload, timeout=20)
        if response.status_code == 200:
            return response.json()
        logger.error(
            "Prediction request failed: %s - %s", response.status_code, response.text
        )
        raise RuntimeError(response.text)
    except requests.RequestException as exc:
        logger.error("Prediction request error: %s", exc)
        raise RuntimeError("Prediction service is unavailable") from exc


def request_file_prediction(file_name: str, file_bytes: bytes, mime_type: str) -> bytes:
    """Call FastAPI file prediction endpoint and return CSV bytes"""
    url = f"{API_BASE_URL}/predict/file"
    files = {"file": (file_name, file_bytes, mime_type or "application/octet-stream")}
    try:
        response = requests.post(url, files=files, timeout=60)
        if response.status_code == 200:
            return response.content
        logger.error(
            "File prediction request failed: %s - %s", response.status_code, response.text
        )
        raise RuntimeError(response.text)
    except requests.RequestException as exc:
        logger.error("File prediction request error: %s", exc)
        raise RuntimeError("Prediction service is unavailable") from exc


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

    # Check for invalid numeric values
    for feature, value in inputs.items():
        if feature in required_features:
            # Skip if already missing
            if feature in missing_fields:
                continue

            # Validate numeric ranges
            if isinstance(value, (int, float)):
                if value < 0:
                    error_messages.append(f"{feature}: Cannot be negative")

    is_valid = len(missing_fields) == 0 and len(error_messages) == 0

    return is_valid, missing_fields, error_messages


def create_input_form(config):
    """Create the input form based on configured features"""

    st.markdown(
        '<div class="main-header">🏠 Housing Price Predictor</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Enter property details to predict the sale price</div>',
        unsafe_allow_html=True,
    )

    # Get features from config
    numeric_features = config.features.numeric
    categorical_features = config.features.categorical

    # Initialize session state for inputs
    if "inputs" not in st.session_state:
        st.session_state.inputs = {}

    # Create tabs for better organization
    tab1, tab2 = st.tabs(["📊 Property Features", "📍 Location & Style"])

    inputs = {}

    with tab1:
        st.subheader("Numerical Features")

        # Organize numeric features into columns
        col1, col2, col3 = st.columns(3)

        # Group features logically
        size_features = [
            "Lot Area",
            "Total Bsmt SF",
            "1st Flr SF",
            "2nd Flr SF",
            "Gr Liv Area",
            "Garage Area",
        ]
        quality_features = ["Overall Qual", "Overall Cond"]
        year_features = ["Year Built", "Year Remod/Add"]
        room_features = [
            "Bedroom AbvGr",
            "Full Bath",
            "Half Bath",
            "TotRms AbvGrd",
            "Fireplaces",
            "Garage Cars",
        ]

        with col1:
            st.markdown("**Size & Area**")
            for feature in size_features:
                if feature in numeric_features:
                    inputs[feature] = st.number_input(
                        feature,
                        min_value=0.0,
                        value=None,
                        step=10.0,
                        help=(
                            f"Enter the {feature.lower()} in square feet"
                            if "SF" in feature or "Area" in feature
                            else f"Enter {feature.lower()}"
                        ),
                        key=f"input_{feature}",
                    )

        with col2:
            st.markdown("**Quality & Year**")
            for feature in quality_features:
                if feature in numeric_features:
                    inputs[feature] = st.number_input(
                        feature,
                        min_value=1,
                        max_value=10,
                        value=None,
                        step=1,
                        help=f"Rate from 1 (poor) to 10 (excellent)",
                        key=f"input_{feature}",
                    )

            for feature in year_features:
                if feature in numeric_features:
                    inputs[feature] = st.number_input(
                        feature,
                        min_value=1800,
                        max_value=2026,
                        value=None,
                        step=1,
                        help=f"Enter the {feature.lower()}",
                        key=f"input_{feature}",
                    )

        with col3:
            st.markdown("**Rooms & Facilities**")
            for feature in room_features:
                if feature in numeric_features:
                    inputs[feature] = st.number_input(
                        feature,
                        min_value=0,
                        max_value=20,
                        value=None,
                        step=1,
                        help=f"Number of {feature.lower()}",
                        key=f"input_{feature}",
                    )

    with tab2:
        st.subheader("Categorical Features")

        col1, col2 = st.columns(2)

        # Define options for categorical features
        categorical_options = {
            "Neighborhood": [
                "Blmngtn",
                "Blueste",
                "BrDale",
                "BrkSide",
                "ClearCr",
                "CollgCr",
                "Crawfor",
                "Edwards",
                "Gilbert",
                "Greens",
                "GrnHill",
                "IDOTRR",
                "Landmrk",
                "MeadowV",
                "Mitchel",
                "NAmes",
                "NPkVill",
                "NWAmes",
                "NoRidge",
                "NridgHt",
                "OldTown",
                "SWISU",
                "Sawyer",
                "SawyerW",
                "Somerst",
                "StoneBr",
                "Timber",
                "Veenker",
            ],
            "MS Zoning": ["A (agr)", "C (all)", "FV", "RH", "RL", "RM"],
            "Bldg Type": ["1Fam", "2fmCon", "Duplex", "Twnhs", "TwnhsE"],
            "House Style": [
                "1.5Fin",
                "1.5Unf",
                "1Story",
                "2.5Fin",
                "2.5Unf",
                "2Story",
                "SFoyer",
                "SLvl",
            ],
            "Foundation": ["BrkTil", "CBlock", "PConc", "Slab", "Stone", "Wood"],
            "Central Air": ["N", "Y"],
            "Garage Type": [
                "2Types",
                "Attchd",
                "Basment",
                "BuiltIn",
                "CarPort",
                "Detchd",
                "nan",
            ],
        }

        with col1:
            for i, feature in enumerate(categorical_features[:4]):
                if feature in categorical_options:
                    inputs[feature] = st.selectbox(
                        feature,
                        options=[""] + categorical_options[feature],
                        help=f"Select {feature.lower()}",
                        key=f"input_{feature}",
                    )
                    # Convert empty string to None
                    if inputs[feature] == "":
                        inputs[feature] = None

        with col2:
            for i, feature in enumerate(categorical_features[4:]):
                if feature in categorical_options:
                    inputs[feature] = st.selectbox(
                        feature,
                        options=[""] + categorical_options[feature],
                        help=f"Select {feature.lower()}",
                        key=f"input_{feature}",
                    )
                    # Convert empty string to None
                    if inputs[feature] == "":
                        inputs[feature] = None

    return inputs, numeric_features + categorical_features


def display_prediction_results(prediction, lower, upper, top_features):
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
                    "95%",
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
        • We are 95% confident the actual price will be between <strong>${lower:,.0f}</strong> and <strong>${upper:,.0f}</strong><br>
        • The top 5 features shown above had the most influence on this prediction<br>
        • The margin of error is <strong>±${margin:,.0f}</strong> ({margin_pct:.1f}%)
    </div>
    """,
        unsafe_allow_html=True,
    )


def build_feedback_record(
    inputs: dict,
    prediction_id: str,
    prediction: float,
    lower: float,
    upper: float,
    agree: bool,
    suggested_min: Optional[float],
    suggested_max: Optional[float],
) -> dict:
    """Build a feedback record for storage"""
    return {
        "feedback_id": str(uuid.uuid4()),
        "prediction_id": prediction_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "agree_with_prediction": agree,
        "predicted_price": float(prediction),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "suggested_min": suggested_min,
        "suggested_max": suggested_max,
        "input_features": json.dumps(inputs),
    }


def render_feedback_form(inputs: dict, result: dict) -> None:
    """Render feedback collection UI and persist feedback"""
    st.markdown("## 📝 Feedback")
    st.markdown(
        "Do you agree with the prediction? Your feedback helps improve the model."
    )

    agree_choice = st.radio(
        "Do you agree with the prediction?",
        options=["Yes, I agree", "No, I disagree"],
        horizontal=True,
        key="feedback_agree_choice",
    )

    suggested_min = None
    suggested_max = None

    if agree_choice == "No, I disagree":
        st.markdown("What price range do you believe is more accurate?")
        col1, col2 = st.columns(2)
        with col1:
            suggested_min = st.number_input(
                "Suggested minimum price",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                key="suggested_min_price",
            )
        with col2:
            suggested_max = st.number_input(
                "Suggested maximum price",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                key="suggested_max_price",
            )

    if st.button("Submit feedback"):
        agree = agree_choice == "Yes, I agree"

        if not agree:
            if suggested_min is None or suggested_max is None:
                st.error("Please provide a suggested price range.")
                return
            if suggested_min <= 0 or suggested_max <= 0:
                st.error("Suggested range must be greater than 0.")
                return
            if suggested_min > suggested_max:
                st.error("Suggested minimum must be less than or equal to maximum.")
                return

        record = build_feedback_record(
            inputs=inputs,
            prediction_id=result["prediction_id"],
            prediction=result["prediction"],
            lower=result["lower"],
            upper=result["upper"],
            agree=agree,
            suggested_min=suggested_min if not agree else None,
            suggested_max=suggested_max if not agree else None,
        )
        save_feedback_record(record)
        st.success("✅ Thanks! Your feedback has been recorded.")


def main():
    """Main application logic"""

    # Load pipeline and config
    with st.spinner("Loading prediction model..."):
        config = load_config()

    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1946/1946488.png", width=100)
        st.title("About")
        st.markdown(
            """
        This application uses a **Gradient Boosting Regressor** model 
        trained on the Ames Housing dataset to predict house prices.
        
        **Model Performance:**
        - R² Score: 0.917
        - RMSE: $25,793
        - MAE: $15,819
        
        **Instructions:**
        1. Fill in all required property details
        2. Click "Predict Price" button
        3. Review prediction with confidence interval
        4. Check which features influenced the prediction most
        """
        )

        st.markdown("---")
        st.markdown("**Model Info:**")
        model_info = fetch_model_info()
        if model_info:
            st.info(f"Model: {model_info.get('model_type', 'unknown')}")
            features = model_info.get("features", {}).get("count")
            if features is not None:
                st.info(f"Features: {features}")
        else:
            st.warning("Model info unavailable. Check API connectivity.")

        st.markdown("---")
        st.markdown("**Need Help?**")
        st.markdown(
            """
        - All fields are required
        - Numerical values must be non-negative
        - Quality ratings are 1-10
        - Years should be between 1800-2026
        """
        )

    # Main content
    inputs, required_features = create_input_form(config)

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
                    payload = build_api_payload(inputs)
                    response = request_prediction(payload)

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
                    st.session_state.last_inputs = inputs

                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    logger.error(f"Prediction error: {str(e)}", exc_info=True)

    if "last_result" in st.session_state and "last_inputs" in st.session_state:
        result = st.session_state.last_result
        last_inputs = st.session_state.last_inputs

        # Display results
        display_prediction_results(
            result["prediction"],
            result["lower"],
            result["upper"],
            result["top_features"],
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

        # Feedback collection
        st.markdown("---")
        render_feedback_form(last_inputs, result)

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
