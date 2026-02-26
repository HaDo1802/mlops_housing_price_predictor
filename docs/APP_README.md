# 🏠 Housing Price Predictor - Streamlit Web Application

A production-ready web interface for predicting housing prices using machine learning.

![App Preview](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)

## 📋 Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Using the Application](#using-the-application)
- [Application Structure](#application-structure)
- [Troubleshooting](#troubleshooting)

## ✨ Features

### User Interface
- **Interactive Input Forms**: Organized tabs for numerical and categorical features
- **Smart Validation**: Real-time validation with helpful error messages
- **Professional Design**: Clean, modern UI with custom styling

### Prediction Features
- **Price Prediction**: Accurate house price estimates using Gradient Boosting model
- **Uncertainty Quantification**: 95% confidence intervals for predictions
- **Feature Importance**: Visual display of top 5 most influential features
- **Interactive Charts**: Dynamic visualizations using Plotly

### Additional Capabilities
- **Results Download**: Export predictions to CSV format
- **Model Information**: View model performance metrics in sidebar
- **Helpful Tooltips**: Contextual help for each input field
- **Error Handling**: Comprehensive error messages and validation

## 🔧 Prerequisites

Before running the application, ensure you have:

1. **Python 3.9 or higher** installed
2. **Trained Model Artifacts** in `models/production/`:
   - `model.pkl`
   - `preprocessor.pkl`
   - `metadata.json`
   - `config.yaml`
3. **Configuration File** in `conf/config.yaml`

## 📦 Installation

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <your-repo-url>
cd housing_price_predictor

# Or download and extract the ZIP file
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements_updated.txt

# Or install manually:
pip install streamlit==1.29.0 plotly==5.18.0 pandas==2.0.3 numpy==1.24.4 scikit-learn==1.3.2
```

## 🚀 Running the App

### Standard Method

```bash
streamlit run serving/app/streamlit_app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

### Custom Port

```bash
streamlit run serving/app/streamlit_app.py --server.port 8080
```

### Run Without Auto-Opening Browser

```bash
streamlit run serving/app/streamlit_app.py --server.headless true
```

### For Production Deployment

```bash
streamlit run serving/app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

## 📖 Using the Application

### Step-by-Step Guide

#### 1. **Launch the Application**
   - Run `streamlit run serving/app/streamlit_app.py`
   - Wait for the browser to open automatically
   - Or navigate to `http://localhost:8501`

#### 2. **Fill in Property Features** (Tab 1: Property Features)
   
   **Size & Area Fields:**
   - Lot Area (sq ft)
   - Total Basement SF
   - 1st Floor SF
   - 2nd Floor SF
   - Ground Living Area
   - Garage Area

   **Quality & Year Fields:**
   - Overall Quality (1-10 scale)
   - Overall Condition (1-10 scale)
   - Year Built (1800-2026)
   - Year Remodeled/Added

   **Rooms & Facilities:**
   - Number of Bedrooms Above Ground
   - Number of Full Bathrooms
   - Number of Half Bathrooms
   - Total Rooms Above Ground
   - Number of Fireplaces
   - Garage Car Capacity

#### 3. **Select Location & Style** (Tab 2: Location & Style)
   
   Choose from dropdown menus:
   - Neighborhood
   - MS Zoning (zoning classification)
   - Building Type
   - House Style
   - Foundation Type
   - Central Air (Y/N)
   - Garage Type

#### 4. **Make Prediction**
   - Click the **"🔮 Predict Price"** button
   - Wait for validation and processing
   - View comprehensive results

#### 5. **Interpret Results**
   
   The results page shows:
   - **Predicted Price**: Main prediction in large, bold text
   - **Confidence Interval**: Range of likely prices (95% confidence)
   - **Statistics Table**: Detailed breakdown of prediction bounds
   - **Feature Importance Chart**: Top 5 features that influenced the prediction
   - **Price Range Visualization**: Interactive chart showing confidence interval
   - **Interpretation Guide**: How to understand the results

#### 6. **Download Results** (Optional)
   - Click **"📥 Download Prediction Results"** button
   - Save CSV file with prediction details

### Example Input Values

Here's a sample property to test the app:

```
Property Features Tab:
- Lot Area: 8,450
- Total Bsmt SF: 850
- 1st Flr SF: 850
- 2nd Flr SF: 850
- Gr Liv Area: 1,700
- Garage Area: 500
- Overall Qual: 7
- Overall Cond: 5
- Year Built: 2003
- Year Remod/Add: 2003
- Bedroom AbvGr: 3
- Full Bath: 2
- Half Bath: 1
- TotRms AbvGrd: 8
- Fireplaces: 1
- Garage Cars: 2

Location & Style Tab:
- Neighborhood: CollgCr
- MS Zoning: RL
- Bldg Type: 1Fam
- House Style: 2Story
- Foundation: PConc
- Central Air: Y
- Garage Type: Attchd
```

## 🏗️ Application Structure

```
serving/app/streamlit_app.py
├── Page Configuration
├── Custom CSS Styling
├── Helper Functions
│   ├── load_pipeline()          # Cached model loading
│   ├── load_config()             # Cached config loading
│   ├── validate_inputs()         # Input validation
│   ├── create_input_form()       # Dynamic form generation
│   └── display_prediction_results()  # Results visualization
└── main()                        # Main application logic
```

## 🎨 UI Components

### Header Section
- Title and subtitle
- Eye-catching design with custom CSS

### Input Forms (2 Tabs)
1. **Property Features Tab**
   - Three columns for organized input
   - Grouped by feature type
   - Helpful tooltips

2. **Location & Style Tab**
   - Dropdown selectors
   - Pre-populated options
   - Two-column layout

### Prediction Button
- Centered, prominent design
- Primary color styling
- Clear call-to-action

### Results Display
- **Prediction Box**: Large, styled container with main prediction
- **Statistics Table**: Detailed metrics
- **Feature Importance**: Horizontal bar chart (Plotly)
- **Price Range Chart**: Interactive line chart with shaded confidence interval
- **Interpretation Box**: User-friendly explanation

### Sidebar
- Model information
- Performance metrics
- Help section
- Instructions

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. **"Model files not found"**
```
Error: Required file not found: models/production/model.pkl
```
**Solution**: Ensure you've trained the model first:
```bash
python pipelines/run_training.py
```

#### 2. **"Module not found" errors**
```
ModuleNotFoundError: No module named 'streamlit'
```
**Solution**: Install dependencies:
```bash
pip install -r requirements_updated.txt
```

#### 3. **Port already in use**
```
OSError: [Errno 48] Address already in use
```
**Solution**: Use a different port:
```bash
streamlit run serving/app/streamlit_app.py --server.port 8502
```

#### 4. **Validation errors persist**
- **Check**: All required fields are filled
- **Verify**: Numeric values are non-negative
- **Confirm**: Years are between 1800-2026
- **Ensure**: Quality ratings are 1-10

#### 5. **Prediction fails**
```
Error making prediction: ...
```
**Solution**: 
- Check model artifacts are properly saved
- Verify input data matches training features
- Check logs for detailed error messages

#### 6. **Charts not displaying**
```
Plotly charts show blank space
```
**Solution**: 
```bash
pip install --upgrade plotly
```

### Performance Issues

If the app is slow:

1. **First Load**: Model loading is cached, so first run may be slower
2. **Large Dataset**: Consider using a more powerful machine
3. **Network**: Ensure good internet connection for CDN resources

### Debug Mode

Run in debug mode for detailed logs:
```bash
streamlit run serving/app/streamlit_app.py --logger.level=debug
```

## 📊 Model Information

The application uses a **Gradient Boosting Regressor** with the following characteristics:

- **Model Type**: GradientBoostingRegressor
- **Training Data**: Ames Housing Dataset
- **Features**: 77 features (16 numeric, 7 categorical → one-hot encoded)
- **Test R² Score**: 0.917
- **Test RMSE**: $25,793
- **Test MAE**: $15,819

### Hyperparameters:
```python
{
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'subsample': 0.8
}
```

## 🔐 Security Considerations

For production deployment:

1. **Input Sanitization**: Already implemented
2. **Error Handling**: Comprehensive try-catch blocks
3. **Environment Variables**: Use for sensitive configurations
4. **HTTPS**: Deploy behind reverse proxy with SSL
5. **Authentication**: Add if needed (e.g., Streamlit Cloud auth)

## 🚀 Deployment Options

### Local Development
```bash
streamlit run serving/app/streamlit_app.py
```

### Streamlit Cloud (Free)
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy with one click

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements_updated.txt .
RUN pip install -r requirements_updated.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "serving/app/streamlit_app.py"]
```

### Heroku
```bash
heroku create housing-predictor
git push heroku main
```

## 📝 Future Enhancements

Potential improvements:

- [ ] Batch prediction from CSV upload
- [ ] Historical price comparison
- [ ] Similar properties finder
- [ ] Advanced filtering options
- [ ] Export to PDF report
- [ ] Multi-language support
- [ ] Mobile-responsive improvements
- [ ] Real-time market data integration

## 📧 Support

For issues or questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review logs in the terminal
3. Open an issue on GitHub
4. Contact the development team

## 📜 License

This project is part of a MLOps learning portfolio.

---

**Built with ❤️ using Streamlit and Python**
