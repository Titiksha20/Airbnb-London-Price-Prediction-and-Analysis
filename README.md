# Airbnb-London-Price-Prediction-and-Analysis

Machine learning model for predicting optimal nightly prices of Airbnb listings in London using XGBoost and SHAP interpretability.

## Overview

This project develops a production-ready pricing model that predicts Airbnb nightly rates with 67% accuracy (R² = 0.667) and provides actionable business insights through SHAP analysis. The model is deployed as an interactive Streamlit web application.

**Live Demo:** [https://airbnb-london-price-prediction-and-analysis.streamlit.app/]

## Key Results

- **Model Performance:** R² = 0.667, RMSE = £86.40, Overfit = 0.071
- **Dataset:** 60,796 London Airbnb listings (September 2024)
- **Features:** 70 engineered features from property, location, amenities, host, and review data
- **Deployment:** Interactive Streamlit app with predictions and explanations

## Business Insights

**Top Price Drivers (from SHAP analysis):**
- Bedrooms: £40 per additional bedroom (£14,600 annual revenue increase)
- Location: £0.51 per £1 increase in neighborhood median price
- Dishwasher: £16/night (often overlooked, high ROI amenity)
- Distance from center: -£2.16 per km (£788/year penalty per km)
- Bathrooms: £19 per additional bathroom

**Market Segmentation:** Identified 5 distinct market segments ranging from mid-market standard (67.6% of listings, avg £127/night) to luxury large properties (2.8%, avg £583/night).

## Project Structure

```
.
├── raw_data/
│   └── listings.csv.gz                 # Original Inside Airbnb data
│
├── final_data/
│   ├── listings_clean_FINAL.csv        # Cleaned dataset (60,796 listings)
│   ├── features_final.csv              # Engineered features
│   ├── neighbourhood_lookup.csv        # Neighborhood statistics
│   └── cleaning_report_final.txt       # Data cleaning decisions
│
├── eda_plots/                          # Exploratory data analysis visualizations
│
├── models/
│   ├── 1_load_data.py
│   ├── 2_data_cleaning_FINAL.py
│   ├── 3_eda_final_cleaned.py
│   ├── 4_feature_engineering_final.py
│   ├── 5_baseline_modeling.py
│   ├── 6_model_optimization.py
│   └── 7_final_model.py
│
├── pkl/
│   ├── xgb_model.pkl                   # Trained XGBoost model
│   └── model_features.txt              # Feature list
│
├── shap/
│   ├── 1_shap_analysis.py             # Basic SHAP analysis
│   ├── 2_shap_advanced.py             # Interaction matrix & clustering
│   └── [plots and results]
│
├── results/                            # Model performance metrics & visualizations
│
├── streamlit/
│   ├── streamlit_app_enhanced.py      # Main Streamlit application
│   ├── create_neighbourhood_lookup.py  # Data preparation script
│   └── save_model.py                   # Model serialization script
│
├── EXECUTIVE_SUMMARY.md                # One-page business overview
└── README.md
```

## Methodology

### 1. Data Cleaning
- Removed outliers (1st-99th percentile)
- Handled missing values (22.2% of listings have no reviews)
- Standardized data types and formats
- Final dataset: 60,796 listings, 70 features

### 2. Feature Engineering
- **Property features:** Bedrooms, bathrooms, capacity, ratios
- **Location features:** Neighborhood pricing, distance from center, competition intensity
- **Amenities:** 17 binary flags (WiFi, kitchen, gym, pool, etc.)
- **Host features:** Superhost status, response rate, experience
- **Review features:** Scores, recency, volume
- **Interaction features:** Location × amenities, size × quality

### 3. Model Development
Evaluated multiple algorithms:
- Linear Regression (baseline)
- Ridge Regression
- Random Forest
- XGBoost
- Ensemble methods

**Winner:** XGBoost with conservative hyperparameters
- n_estimators: 200
- max_depth: 6
- learning_rate: 0.05
- reg_lambda: 0.5 (L2 regularization)

### 4. Model Validation
- Train/test split: 80/20
- 5-fold cross-validation: R² = 0.655 ± 0.008
- No overfitting (train R² = 0.737, test R² = 0.667)

### 5. Interpretability
SHAP (SHapley Additive exPlanations) analysis:
- Global feature importance
- Individual prediction explanations
- Feature interaction detection
- Market segmentation (5 clusters)

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone repository
git clone https://github.com/Titiksha20/Airbnb-London-Price-Prediction-and-Analysis.git

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
cd streamlit
streamlit run streamlit_app_enhanced.py
```

### Requirements
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
streamlit>=1.28.0
plotly>=5.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
shap>=0.42.0
```

## Usage

### Running the Full Pipeline

```bash
# 1. Load and clean data
python models/1_load_data.py
python models/2_data_cleaning_FINAL.py

# 2. Exploratory data analysis
python models/3_eda_final_cleaned.py

# 3. Feature engineering
python models/4_feature_engineering_final.py

# 4. Model training
python models/5_baseline_modeling.py
python models/6_model_optimization.py
python models/7_final_model.py

# 5. SHAP analysis
python shap/1_shap_analysis.py
python shap/2_shap_advanced.py
```

### Using the Streamlit App

```bash
python streamlit/create_neighbourhood_lookup.py
python streamlit/save_model.py

cd streamlit
streamlit run streamlit_app_enhanced.py
```

The streamlit site provides:
- price predictions
- Confidence intervals
- Market position analysis
- Pricing recommendations
- Similar listings comparison

## Model Performance

The model explains 67% of price variance, with an average prediction error of £86. This meets industry standards for pricing models (typical R² range: 0.65-0.70).


## Technical Stack

- **Languages:** Python 3.9
- **ML Libraries:** XGBoost, Scikit-learn, SHAP
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Deployment:** Streamlit, Streamlit Cloud
- **Version Control:** Git, GitHub

## Data Source

Data from [Inside Airbnb](http://insideairbnb.com/get-the-data/), an independent, non-commercial project that provides Airbnb listing data for research purposes.

**Dataset:** London, United Kingdom 

## License

This project is for educational and portfolio purposes. Data provided by Inside Airbnb under Creative Commons CC0 1.0 Universal (CC0 1.0) Public Domain Dedication.


