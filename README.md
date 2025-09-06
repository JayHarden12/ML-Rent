Nigerian House Rent Price Estimation System
==========================================

A Streamlit app for predicting Nigerian house rent prices using machine learning. It includes interactive prediction, data exploration, location insights, and model performance dashboards.

Features
--------
- Interactive price predictor based on property attributes
- Data analysis dashboard with histograms and comparisons
- Location insights (top/cheapest areas and stats)
- Model performance metrics and feature importance
- Multiple models: Random Forest, Gradient Boosting, Linear Regression

Quick Start
-----------

Prerequisites
- Python 3.9+ (3.11 recommended)
- pip

1) Create and activate a virtual environment (optional but recommended)

Windows (PowerShell):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux (bash):
```
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies
```
pip install -r requirements.txt
```

3) Ensure the dataset is present
- Place `nigeria-rent.csv` in the project root (same folder as `app.py`).
- Expected columns include: `Price`, `Bedrooms`, `Bathrooms`, `Toilets`, `Location`, `Serviced`, `Newly Built`, `Furnished`.

4) Run the app
```
streamlit run app.py
```
Then open the provided URL (usually http://localhost:8501).

Dataset
-------
The app uses `nigeria-rent.csv`, which contains property listings with prices (NGN), attributes, and locations. The preprocessing pipeline:
- Cleans price strings and extracts numeric values
- Extracts numeric counts from text columns (bedrooms/bathrooms/toilets)
- Removes outliers and missing essential values
- Engineers features (total rooms, price per bedroom/room)
- Label‑encodes area names

Project Structure
-----------------
```
app.py                 # Streamlit application UI
data_preprocessor.py   # Data cleaning and feature preparation
ml_models.py           # Model training, selection, and prediction logic
requirements.txt       # Python dependencies
nigeria-rent.csv       # Dataset (not tracked if large)
README.md              # Documentation
```

Technical Details
-----------------
- Split: 80/20 train/test with `train_test_split`
- Models: RandomForestRegressor, GradientBoostingRegressor, LinearRegression
- Metrics: R^2, RMSE, MAE (reported per model)
- Feature importance: available for tree‑based models
- Caching: `@st.cache_data` speeds up reloads

The standalone training script in `ml_models.py` also supports optional hyperparameter tuning with `GridSearchCV` (cv=3).

Usage Guide
-----------
1) Price Predictor
- Set bedrooms, bathrooms, toilets, and amenities
- Select area; the app encodes it to match the model
- View the predicted rent and best model R^2

2) Data Analysis
- Price distribution histogram
- Average price by bedrooms/bathrooms
- Distribution of amenities (serviced/newly built/furnished)

3) Location Insights
- Top 10 most expensive and most affordable areas
- Area statistics table (average price and listing counts)

4) Model Performance
- Per‑model metrics (R^2, RMSE, MAE)
- Feature importance bar chart (if available)
- Best model highlight and brief recommendations

Train and Save a Model (optional)
---------------------------------
Run the standalone trainer to save a model artifact:
```
python ml_models.py
```
This reads `nigeria-rent.csv`, trains models, may tune a tree‑based model, and saves `rent_predictor_model.pkl` with the estimator, scaler, and feature importance. The Streamlit app trains on the fly by default and does not require a pre‑saved model.

Requirements
------------
```
streamlit==1.28.1
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
plotly==5.17.0
seaborn==0.12.2
matplotlib==3.7.2
```

Troubleshooting
---------------
- Dataset not found: verify `nigeria-rent.csv` exists alongside `app.py`.
- Slow first run: models train on startup; subsequent runs are faster due to caching.
- Plotly axis updates: code uses `update_xaxes`/`update_yaxes` for compatibility.
- Info logs like "No runtime found, using MemoryCacheStorageManager" are harmless.

Contributing
------------
1. Fork this repository
2. Create a feature branch
3. Make changes with tests/validation
4. Open a pull request

License
-------
MIT License

Acknowledgments
---------------
- Dataset: Nigerian property rental data
- Libraries: Streamlit, scikit‑learn, Plotly, Pandas, NumPy

—

Built for the Nigerian real estate market.
