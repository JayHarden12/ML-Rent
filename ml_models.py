import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class RentPricePredictor:
    def get_feature_importance(self, model, feature_names):
        """Return feature importances for tree-based models as a pandas Series"""
        if hasattr(model, 'feature_importances_'):
            import pandas as pd
            return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        return None
    def select_best_model(self, results):
        """Select the best model based on R² score"""
        best_r2 = -np.inf
        best_model = None
        best_name = None
        for name, metrics in results.items():
            if metrics['r2'] > best_r2:
                best_r2 = metrics['r2']
                best_model = metrics['model']
                best_name = name
        self.best_model = best_model
        return best_name, best_model
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
            'Linear Regression': LinearRegression()
        }
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all models and evaluate performance"""
        results = {}
        
        # Clean infinite and very large values in training and test data
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        X_train = np.clip(X_train, -1e6, 1e6)
        X_test = np.clip(X_test, -1e6, 1e6)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
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
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.9, 1.0]
            }
            model = GradientBoostingRegressor(random_state=42)
        
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Grid search with cross-validation
        try:
            grid_search = GridSearchCV(
                model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=2
            )
            grid_search.fit(X_train, y_train)
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation R²: {grid_search.best_score_:.4f}")
        except Exception as e:
            print(f"Error during hyperparameter tuning: {str(e)}")
            print(f"Model: {model_name}")
            print(f"Parameter grid: {param_grid}")
            print(f"X_train shape: {getattr(X_train, 'shape', 'N/A')}")
            print(f"y_train shape: {getattr(y_train, 'shape', 'N/A')}")
            return None
        
        return grid_search.best_estimator_
    
    def predict_price(self, features):
        """Make price prediction using the best model"""
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        # Handle scaling if needed
        if hasattr(self.best_model, 'feature_importances_'):  # Tree-based model
            pred = self.best_model.predict(features.reshape(1, -1))
        else:  # Linear model
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            pred = self.best_model.predict(features_scaled)
        # Safely extract scalar prediction
        if hasattr(pred, 'item') and pred.size == 1:
            prediction = pred.item()
        elif isinstance(pred, (list, np.ndarray)) and len(pred) == 1:
            prediction = pred[0]
        else:
            prediction = pred
        
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
        fi = predictor.feature_importance
        try:
            import pandas as pd
            if isinstance(fi, (pd.Series, pd.DataFrame)):
                print(fi.head(10))
            else:
                print(fi)
        except Exception as e:
            print(f"Error printing feature importance: {e}")
    
    # Hyperparameter tuning for best model
    if best_name in ['Random Forest', 'Gradient Boosting']:
        tuned_model = predictor.hyperparameter_tuning(X_train, y_train, best_name)
        if tuned_model is not None:
            try:
                import numpy as np
                y_pred_tuned = tuned_model.predict(X_test)
                y_pred_tuned = np.ravel(y_pred_tuned)
                y_test_1d = np.ravel(y_test)
                tuned_r2 = r2_score(y_test_1d, y_pred_tuned)
                print(f"Tuned model R²: {tuned_r2:.4f}")
                # Compare tuned model against the currently selected best model's R²
                if tuned_r2 > results[best_name]['r2']:
                    predictor.best_model = tuned_model
                    print("Using tuned model as final model")
            except Exception as e:
                print(f"Error in tuned model prediction or metric calculation: {e}")
    
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
