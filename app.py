import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_preprocessor import DataPreprocessor
from ml_models import RentPricePredictor
import re
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Nigerian House Rent Price Estimation System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the dataset using DataPreprocessor"""
    try:
        df = pd.read_csv('nigeria-rent.csv')
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.preprocess_data(df)
        return df_clean, preprocessor
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), None

@st.cache_data
def train_ml_models(df_clean, _preprocessor):
    """Train machine learning models"""
    try:
        X, y, feature_names = _preprocessor.prepare_features(df_clean)
        
        # Initialize predictor
        predictor = RentPricePredictor()
        predictor.initialize_models()
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train models
        results = predictor.train_models(X_train, y_train, X_test, y_test)
        
        # Select best model
        best_name, best_metrics = predictor.select_best_model(results)
        
        return predictor, results, feature_names
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        return None, None, None

def format_price(price):
    """Format price for display"""
    if price >= 1000000:
        return f"₦{price/1000000:.1f}M"
    elif price >= 1000:
        return f"₦{price/1000:.0f}K"
    else:
        return f"₦{price:.0f}"

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 Nigerian House Rent Price Estimation System</h1>', unsafe_allow_html=True)
    
    # Load and preprocess data
    with st.spinner("Loading and preprocessing data..."):
        df_clean, preprocessor = load_and_preprocess_data()
    
    if df_clean.empty or preprocessor is None:
        st.error("Failed to load data. Please check if 'nigeria-rent.csv' exists in the current directory.")
        return
    
    # Train ML models
    with st.spinner("Training machine learning models..."):
        predictor, results, feature_names = train_ml_models(df_clean, preprocessor)
    
    if predictor is None:
        st.error("Failed to train models. Please check the data and try again.")
        return
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Data overview
    st.sidebar.markdown("### 📊 Dataset Overview")
    st.sidebar.metric("Total Properties", len(df_clean))
    st.sidebar.metric("Average Price", format_price(df_clean['Price_Numeric'].mean()))
    st.sidebar.metric("Price Range", f"{format_price(df_clean['Price_Numeric'].min())} - {format_price(df_clean['Price_Numeric'].max())}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Price Predictor", "📈 Data Analysis", "🗺️ Location Insights", "📊 Model Performance"])
    
    with tab1:
        st.header("🏠 Property Rent Price Predictor")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Property Details")
            
            bedrooms = st.slider("Number of Bedrooms", 1, 10, 3)
            bathrooms = st.slider("Number of Bathrooms", 1, 10, 3)
            toilets = st.slider("Number of Toilets", 1, 10, 3)
            
            serviced = st.checkbox("Serviced Apartment")
            newly_built = st.checkbox("Newly Built")
            furnished = st.checkbox("Furnished")
            
            # Location selection
            areas = sorted(df_clean['Area'].unique())
            selected_area = st.selectbox("Select Area", areas)
            
        with col2:
            st.subheader("Price Prediction")
            
            # Prepare input features
            input_features = np.array([
                bedrooms, bathrooms, toilets, int(serviced), 
                int(newly_built), int(furnished), 
                bedrooms + bathrooms + toilets,  # Total rooms
                bedrooms + bathrooms + toilets,  # Placeholder for price per bedroom
                bedrooms + bathrooms + toilets   # Placeholder for price per room
            ])
            
            # Add area encoding if available
            if 'Area_Encoded' in df_clean.columns and selected_area in preprocessor.label_encoders['Area'].classes_:
                area_encoded = preprocessor.label_encoders['Area'].transform([selected_area])[0]
                input_features = np.append(input_features, area_encoded)
            else:
                input_features = np.append(input_features, 0)  # Default area encoding
            
            # Make prediction
            try:
                prediction = predictor.predict_price(input_features)
                
                # Display prediction
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f"### 🎯 Predicted Rent Price")
                st.markdown(f"**Estimated Price:** {format_price(prediction)}")
                st.markdown(f"**Area:** {selected_area}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show model confidence
                if results:
                    best_r2 = max([metrics['r2'] for metrics in results.values()])
                    st.info(f"Model R² Score: {best_r2:.3f}")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please try adjusting the input parameters.")
    
    with tab2:
        st.header("📈 Data Analysis Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution
            fig_price = px.histogram(df_clean, x='Price_Numeric', nbins=50, 
                                   title="Price Distribution",
                                   labels={'Price_Numeric': 'Price (₦)', 'count': 'Number of Properties'})
            fig_price.update_layout(xaxis_title="Price (₦)", yaxis_title="Count")
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Bedroom vs Price
            bedroom_price = df_clean.groupby('Bedrooms_Numeric')['Price_Numeric'].mean().reset_index()
            fig_bedroom = px.bar(bedroom_price, x='Bedrooms_Numeric', y='Price_Numeric',
                               title="Average Price by Number of Bedrooms",
                               labels={'Bedrooms_Numeric': 'Number of Bedrooms', 'Price_Numeric': 'Average Price (₦)'})
            st.plotly_chart(fig_bedroom, use_container_width=True)
        
        with col2:
            # Property features distribution
            feature_cols = ['Serviced', 'Newly Built', 'Furnished']
            feature_counts = df_clean[feature_cols].sum()
            
            fig_features = px.pie(values=feature_counts.values, names=feature_counts.index,
                                title="Property Features Distribution")
            st.plotly_chart(fig_features, use_container_width=True)
            
            # Bathroom vs Price
            bathroom_price = df_clean.groupby('Bathrooms_Numeric')['Price_Numeric'].mean().reset_index()
            fig_bathroom = px.bar(bathroom_price, x='Bathrooms_Numeric', y='Price_Numeric',
                                title="Average Price by Number of Bathrooms",
                                labels={'Bathrooms_Numeric': 'Number of Bathrooms', 'Price_Numeric': 'Average Price (₦)'})
            st.plotly_chart(fig_bathroom, use_container_width=True)
    
    with tab3:
        st.header("🗺️ Location Insights")
        
        # Top areas by average price
        area_stats = df_clean.groupby('Area').agg({
            'Price_Numeric': ['mean', 'count']
        }).round(0)
        area_stats.columns = ['Average_Price', 'Property_Count']
        area_stats = area_stats[area_stats['Property_Count'] >= 5].sort_values('Average_Price', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Most Expensive Areas")
            top_areas = area_stats.head(10)
            fig_top = px.bar(top_areas, x=top_areas.index, y='Average_Price',
                           title="Top 10 Most Expensive Areas",
                           labels={'index': 'Area', 'Average_Price': 'Average Price (₦)'})
            fig_top.update_xaxes(tickangle=45)
            st.plotly_chart(fig_top, use_container_width=True)
        
        with col2:
            st.subheader("💰 Most Affordable Areas")
            bottom_areas = area_stats.tail(10)
            fig_bottom = px.bar(bottom_areas, x=bottom_areas.index, y='Average_Price',
                              title="Top 10 Most Affordable Areas",
                              labels={'index': 'Area', 'Average_Price': 'Average Price (₦)'})
            fig_bottom.update_xaxes(tickangle=45)
            st.plotly_chart(fig_bottom, use_container_width=True)
        
        # Area statistics table
        st.subheader("📊 Area Statistics")
        st.dataframe(area_stats.head(20), use_container_width=True)
    
    with tab4:
        st.header("📊 Model Performance Analysis")
        
        if results and feature_names:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Model Accuracy")
                
                # Display model metrics
                for name, metrics in results.items():
                    st.metric(f"{name} R² Score", f"{metrics['r2']:.3f}")
                    st.metric(f"{name} RMSE", f"{metrics['rmse']:,.0f}")
                    st.metric(f"{name} MAE", f"{metrics['mae']:,.0f}")
                    st.write("---")
                
                # Feature importance for best model
                if predictor.feature_importance is not None:
                    st.subheader("🔍 Feature Importance")
                    fig_importance = px.bar(
                        predictor.feature_importance.head(10).to_frame(name='importance').reset_index().rename(columns={'index': 'feature'}), 
                        x='importance', 
                        y='feature',
                        orientation='h',
                        title="Top 10 Most Important Features",
                        labels={'importance': 'Importance', 'feature': 'Feature'}
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
            
            with col2:
                st.subheader("📈 Model Comparison")
                
                # Model comparison chart
                model_names = list(results.keys())
                r2_scores = [metrics['r2'] for metrics in results.values()]
                
                fig_comparison = px.bar(
                    x=model_names, 
                    y=r2_scores,
                    title="Model R² Score Comparison",
                    labels={'x': 'Model', 'y': 'R² Score'}
                )
                fig_comparison.update_xaxes(tickangle=45)
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Best model info
                best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
                best_r2 = results[best_model_name]['r2']
                
                st.subheader("🏆 Best Performing Model")
                st.success(f"**{best_model_name}** with R² = {best_r2:.3f}")
                
                # Model recommendations
                st.subheader("💡 Model Recommendations")
                if best_r2 > 0.7:
                    st.success("✅ Excellent model performance! The model can make reliable predictions.")
                elif best_r2 > 0.5:
                    st.warning("⚠️ Good model performance. Consider feature engineering for better results.")
                else:
                    st.error("❌ Poor model performance. Consider data quality and feature selection.")
                
                # Feature names display
                st.subheader("📋 Model Features")
                st.write("Features used in the model:")
                for i, feature in enumerate(feature_names, 1):
                    st.write(f"{i}. {feature}")
        else:
            st.error("Model performance data not available. Please check if models were trained successfully.")

if __name__ == "__main__":
    main()
