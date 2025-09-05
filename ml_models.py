import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class RentPricePredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_importance = None
        
    def initialize_models(self):
        """Initialize various ML models"""
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Support Vector Regression': SVR(kernel='rbf')
        }
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all models and evaluate performance"""
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Scale features for models that need it
            if name in ['Ridge Regression', 'Lasso Regression', 'Support Vector Regression']:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"{name} - R²: {r2:.4f}, RMSE: {rmse:,.0f}, MAE: {mae:,.0f}")
        
        return results
    
    def select_best_model(self, results):
        """Select the best model based on R² score"""
        best_r2 = -np.inf
        best_model_name = None
        
        for name, metrics in results.items():
            if metrics['r2'] > best_r2:
                best_r2 = metrics['r2']
                best_model_name = name
        
        self.best_model = results[best_model_name]['model']
        print(f"\nBest model: {best_model_name} with R² = {best_r2:.4f}")
        
        return best_model_name, results[best_model_name]
    
    def get_feature_importance(self, model, feature_names):
        """Get feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return feature_importance
        return None
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='Random Forest'):
        """Perform hyperparameter tuning for the best model"""
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42)
        
        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = GradientBoostingRegressor(random_state=42)
        
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation R²: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def predict_price(self, features):
        """Make price prediction using the best model"""
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        # Handle scaling if needed
        if hasattr(self.best_model, 'feature_importances_'):  # Tree-based model
            prediction = self.best_model.predict(features.reshape(1, -1))[0]
        else:  # Linear model
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            prediction = self.best_model.predict(features_scaled)[0]
        
        return max(0, prediction)  # Ensure non-negative prediction
    
    def save_model(self, filepath='rent_predictor_model.pkl'):
        """Save the trained model"""
        if self.best_model is not None:
            joblib.dump({
                'model': self.best_model,
                'scaler': self.scaler,
                'feature_importance': self.feature_importance
            }, filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save")
    
    def load_model(self, filepath='rent_predictor_model.pkl'):
        """Load a trained model"""
        try:
            model_data = joblib.load(filepath)
            self.best_model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_importance = model_data.get('feature_importance', None)
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

def train_rent_predictor(X, y, feature_names):
    """Main function to train the rent price predictor"""
    print("Starting rent price prediction model training...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Initialize predictor
    predictor = RentPricePredictor()
    predictor.initialize_models()
    
    # Train models
    results = predictor.train_models(X_train, y_train, X_test, y_test)
    
    # Select best model
    best_name, best_metrics = predictor.select_best_model(results)
    
    # Get feature importance
    if hasattr(predictor.best_model, 'feature_importances_'):
        predictor.feature_importance = predictor.get_feature_importance(
            predictor.best_model, feature_names
        )
        print("\nFeature Importance:")
        print(predictor.feature_importance.head(10))
    
    # Hyperparameter tuning for best model
    if best_name in ['Random Forest', 'Gradient Boosting']:
        tuned_model = predictor.hyperparameter_tuning(X_train, y_train, best_name)
        if tuned_model is not None:
            # Evaluate tuned model
            if best_name in ['Ridge Regression', 'Lasso Regression', 'Support Vector Regression']:
                X_test_scaled = predictor.scaler.transform(X_test)
                y_pred_tuned = tuned_model.predict(X_test_scaled)
            else:
                y_pred_tuned = tuned_model.predict(X_test)
            
            tuned_r2 = r2_score(y_test, y_pred_tuned)
            print(f"Tuned model R²: {tuned_r2:.4f}")
            
            if tuned_r2 > best_metrics['r2']:
                predictor.best_model = tuned_model
                print("Using tuned model as final model")
    
    # Save model
    predictor.save_model()
    
    return predictor, results

def main():
    """Test the ML models"""
    try:
        # Load preprocessed data
        from data_preprocessor import DataPreprocessor
        
        print("Loading and preprocessing data...")
        df = pd.read_csv('nigeria-rent.csv')
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.preprocess_data(df)
        
        # Prepare features
        X, y, feature_names = preprocessor.prepare_features(df_clean)
        
        # Train models
        predictor, results = train_rent_predictor(X, y, feature_names)
        
        return predictor, results, df_clean
        
    except Exception as e:
        print(f"Error in ML training: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    predictor, results, df_clean = main()
