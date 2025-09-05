import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def clean_price(self, price_str):
        """Clean and extract numeric value from price string"""
        if pd.isna(price_str):
            return np.nan
        
        # Remove commas and extract numeric value
        price_str = str(price_str).replace(',', '')
        
        # Extract numbers from string
        numbers = re.findall(r'\d+', price_str)
        if numbers:
            return float(''.join(numbers))
        return np.nan
    
    def extract_numeric_feature(self, feature_str):
        """Extract numeric value from feature string"""
        if pd.isna(feature_str):
            return np.nan
        
        numbers = re.findall(r'\d+', str(feature_str))
        return int(numbers[0]) if numbers else np.nan
    
    def extract_area(self, location_str):
        """Extract area from location string"""
        if pd.isna(location_str):
            return 'Unknown'
        
        # Split by comma and take the last part (usually the area)
        parts = str(location_str).split(',')
        if len(parts) >= 2:
            return parts[-1].strip()
        return str(location_str).strip()
    
    def preprocess_data(self, df):
        """Main preprocessing function"""
        print("Starting data preprocessing...")
        
        # Clean price data
        print("Cleaning price data...")
        df['Price_Numeric'] = df['Price'].apply(self.clean_price)
        
        # Clean bedroom data
        print("Cleaning bedroom data...")
        df['Bedrooms_Numeric'] = df['Bedrooms'].apply(self.extract_numeric_feature)
        
        # Clean bathroom data
        print("Cleaning bathroom data...")
        df['Bathrooms_Numeric'] = df['Bathrooms'].apply(self.extract_numeric_feature)
        
        # Clean toilet data
        print("Cleaning toilet data...")
        df['Toilets_Numeric'] = df['Toilets'].apply(self.extract_numeric_feature)
        
        # Extract location information
        print("Extracting location information...")
        df['Area'] = df['Location'].apply(self.extract_area)
        
        # Remove rows with missing essential data
        print("Removing rows with missing data...")
        initial_count = len(df)
        df_clean = df.dropna(subset=['Price_Numeric', 'Bedrooms_Numeric', 'Bathrooms_Numeric'])
        removed_count = initial_count - len(df_clean)
        print(f"Removed {removed_count} rows with missing essential data")
        
        # Remove outliers (prices above 50M and below 100K)
        print("Removing outliers...")
        before_outlier_removal = len(df_clean)
        df_clean = df_clean[(df_clean['Price_Numeric'] >= 100000) & (df_clean['Price_Numeric'] <= 50000000)]
        after_outlier_removal = len(df_clean)
        print(f"Removed {before_outlier_removal - after_outlier_removal} outliers")
        
        # Create additional features
        print("Creating additional features...")
        df_clean['Total_Rooms'] = df_clean['Bedrooms_Numeric'] + df_clean['Bathrooms_Numeric'] + df_clean['Toilets_Numeric']
        df_clean['Price_Per_Bedroom'] = df_clean['Price_Numeric'] / df_clean['Bedrooms_Numeric']
        df_clean['Price_Per_Room'] = df_clean['Price_Numeric'] / df_clean['Total_Rooms']
        
        # Handle categorical features
        print("Encoding categorical features...")
        categorical_features = ['Area']
        for feature in categorical_features:
            if feature in df_clean.columns:
                le = LabelEncoder()
                df_clean[f'{feature}_Encoded'] = le.fit_transform(df_clean[feature].astype(str))
                self.label_encoders[feature] = le
        
        print(f"Data preprocessing completed. Final dataset shape: {df_clean.shape}")
        return df_clean
    
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        # Select features for modeling
        feature_columns = [
            'Bedrooms_Numeric', 'Bathrooms_Numeric', 'Toilets_Numeric',
            'Serviced', 'Newly Built', 'Furnished', 'Total_Rooms',
            'Price_Per_Bedroom', 'Price_Per_Room'
        ]
        
        # Add encoded area if available
        if 'Area_Encoded' in df.columns:
            feature_columns.append('Area_Encoded')
        
        # Create feature matrix
        X = df[feature_columns].fillna(0)
        y = df['Price_Numeric']
        
        return X, y, feature_columns
    
    def get_data_summary(self, df):
        """Get summary statistics of the dataset"""
        summary = {
            'total_properties': len(df),
            'average_price': df['Price_Numeric'].mean(),
            'median_price': df['Price_Numeric'].median(),
            'min_price': df['Price_Numeric'].min(),
            'max_price': df['Price_Numeric'].max(),
            'price_std': df['Price_Numeric'].std(),
            'unique_areas': df['Area'].nunique(),
            'bedroom_distribution': df['Bedrooms_Numeric'].value_counts().to_dict(),
            'bathroom_distribution': df['Bathrooms_Numeric'].value_counts().to_dict(),
            'serviced_count': df['Serviced'].sum(),
            'newly_built_count': df['Newly Built'].sum(),
            'furnished_count': df['Furnished'].sum()
        }
        return summary

def main():
    """Test the data preprocessor"""
    try:
        # Load data
        print("Loading data...")
        df = pd.read_csv('nigeria-rent.csv')
        print(f"Original dataset shape: {df.shape}")
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Preprocess data
        df_clean = preprocessor.preprocess_data(df)
        
        # Get summary
        summary = preprocessor.get_data_summary(df_clean)
        print("\nDataset Summary:")
        print(f"Total Properties: {summary['total_properties']:,}")
        print(f"Average Price: ₦{summary['average_price']:,.0f}")
        print(f"Price Range: ₦{summary['min_price']:,.0f} - ₦{summary['max_price']:,.0f}")
        print(f"Unique Areas: {summary['unique_areas']}")
        
        # Prepare features
        X, y, feature_columns = preprocessor.prepare_features(df_clean)
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Features: {feature_columns}")
        
        return df_clean, preprocessor
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None, None

if __name__ == "__main__":
    df_clean, preprocessor = main()
