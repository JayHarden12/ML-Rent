# 🏠 Nigerian House Rent Price Estimation System

A comprehensive web application built with Python and Streamlit for predicting house rent prices in Nigeria using machine learning models.

## 🌟 Features

- **Interactive Price Predictor**: Get instant rent price estimates based on property features
- **Data Analysis Dashboard**: Visualize rent trends and property distributions
- **Location Insights**: Compare prices across different areas in Nigeria
- **Model Performance Analysis**: View detailed model metrics and feature importance
- **Multiple ML Models**: Random Forest, Gradient Boosting, Linear Regression, and more

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone or download the project files**
   ```bash
   # Ensure you have the following files in your directory:
   # - app.py
   # - data_preprocessor.py
   # - ml_models.py
   # - requirements.txt
   # - nigeria-rent.csv
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, copy the URL from the terminal

## 📊 Dataset

The application uses the `nigeria-rent.csv` dataset containing:
- **98,000+ property listings** from various areas in Nigeria
- **Property features**: Bedrooms, bathrooms, toilets, serviced status, etc.
- **Location data**: Areas across Nigeria including Lagos, Abuja, and more
- **Price information**: Annual rent prices in Nigerian Naira (₦)

## 🏗️ Project Structure

```
├── app.py                 # Main Streamlit application
├── data_preprocessor.py   # Data cleaning and preprocessing
├── ml_models.py          # Machine learning models and training
├── requirements.txt      # Python dependencies
├── nigeria-rent.csv     # Dataset
└── README.md            # This file
```

## 🔧 Technical Details

### Data Preprocessing
- **Price cleaning**: Extracts numeric values from price strings
- **Feature extraction**: Converts text features to numeric values
- **Outlier removal**: Filters unrealistic price ranges
- **Feature engineering**: Creates additional features like total rooms, price per bedroom

### Machine Learning Models
- **Random Forest Regressor**: Handles non-linear relationships
- **Gradient Boosting**: Advanced ensemble method
- **Linear Regression**: Simple baseline model
- **Ridge/Lasso Regression**: Regularized linear models
- **Support Vector Regression**: Kernel-based regression

### Model Selection
- **Cross-validation**: 5-fold cross-validation for robust evaluation
- **Hyperparameter tuning**: Grid search for optimal parameters
- **Performance metrics**: R² score, RMSE, MAE
- **Feature importance**: Identifies most influential features

## 📈 Usage Guide

### 1. Price Predictor Tab
- Adjust property features using sliders and checkboxes
- Select the area from the dropdown
- View instant price predictions
- See model confidence scores

### 2. Data Analysis Tab
- Explore price distributions
- Analyze bedroom/bathroom vs price relationships
- View property feature distributions

### 3. Location Insights Tab
- Compare average prices across areas
- Find most expensive and affordable areas
- View detailed area statistics

### 4. Model Performance Tab
- Compare different ML models
- View feature importance rankings
- Understand model accuracy and recommendations

## 🎯 Key Features Explained

### Price Prediction
The system uses multiple machine learning models to predict rent prices based on:
- Number of bedrooms, bathrooms, and toilets
- Property amenities (serviced, newly built, furnished)
- Location (area encoding)
- Derived features (total rooms, price per room)

### Data Visualization
- **Interactive charts** using Plotly
- **Real-time updates** based on user selections
- **Responsive design** for different screen sizes
- **Professional styling** with custom CSS

### Model Performance
- **Comprehensive evaluation** with multiple metrics
- **Feature importance analysis** for model interpretability
- **Model comparison** to identify best performers
- **Performance recommendations** based on R² scores

## 🔍 Model Performance

The application typically achieves:
- **R² Score**: 0.6-0.8 (depending on data quality)
- **RMSE**: Varies by price range
- **Best performing model**: Usually Random Forest or Gradient Boosting

## 🛠️ Customization

### Adding New Features
1. Modify `data_preprocessor.py` to include new features
2. Update feature list in `ml_models.py`
3. Add UI controls in `app.py`

### Improving Model Performance
1. **Feature Engineering**: Add more derived features
2. **Data Quality**: Improve data cleaning and outlier detection
3. **Model Tuning**: Adjust hyperparameters
4. **Ensemble Methods**: Combine multiple models

### Styling
- Modify CSS in the `st.markdown()` section of `app.py`
- Update colors, fonts, and layout as needed

## 📝 Requirements

```
streamlit==1.28.1
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
plotly==5.17.0
seaborn==0.12.2
matplotlib==3.7.2
```

## 🚨 Troubleshooting

### Common Issues

1. **Module not found errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset not found**
   - Ensure `nigeria-rent.csv` is in the same directory as `app.py`

3. **Memory issues with large dataset**
   - Reduce dataset size or use data sampling
   - Increase system memory

4. **Slow loading**
   - The app caches data and models for faster subsequent loads
   - First run may take longer due to model training

### Performance Optimization

1. **Enable caching**: The app uses `@st.cache_data` for optimal performance
2. **Reduce data size**: Consider sampling for development
3. **Model persistence**: Save trained models to avoid retraining

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset: Nigerian property rental data
- Libraries: Streamlit, scikit-learn, Plotly, Pandas
- Community: Open source contributors

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code comments
3. Create an issue in the repository

---

**Built with ❤️ for the Nigerian real estate market**
