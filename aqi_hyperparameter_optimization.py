"""
🔧 AQI Model Hyperparameter Optimization with Optuna
===================================================

This script performs comprehensive hyperparameter optimization for the AQI prediction model
using Optuna's advanced optimization algorithms. It optimizes LightGBM parameters to improve
model performance beyond the current RMSE: 2.57, R²: 0.997 baseline.

Features:
- Multi-objective optimization (RMSE + R² + MAPE)
- Cross-validation based evaluation
- Pruning for efficient optimization
- Best parameters integration
- Performance comparison
"""

import optuna
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import hopsworks
import warnings
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# Get Hopsworks credentials
HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')
HOPSWORKS_PROJECT = os.getenv('HOPSWORKS_PROJECT', 'aqi-prediction-karachi')

class AQIHyperparameterOptimizer:
    def __init__(self):
        self.project = None
        self.fs = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.best_params = None
        self.study = None
        
    def connect_to_hopsworks(self):
        """Connect to Hopsworks and get feature store"""
        print("🔗 Connecting to Hopsworks...")
        
        # Check if credentials are available
        if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT:
            print("❌ Hopsworks credentials not found. Please set HOPSWORKS_API_KEY and HOPSWORKS_PROJECT environment variables.")
            return False
        
        try:
            self.project = hopsworks.login(
                api_key_value=HOPSWORKS_API_KEY,
                project=HOPSWORKS_PROJECT
            )
            self.fs = self.project.get_feature_store()
            print("✅ Connected to Hopsworks successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Hopsworks: {e}")
            return False
    
    def load_and_prepare_data(self):
        """Load data from Hopsworks and prepare for optimization"""
        print("📊 Loading and preparing data...")
        
        try:
            # Get the latest version dynamically
            feature_group_name = "karachifeatures"
            latest_version = self._get_latest_feature_group_version(feature_group_name)
            
            if latest_version == 0:
                raise RuntimeError("❌ No feature group versions found in Hopsworks")
            
            print(f"📊 Loading feature group '{feature_group_name}' version {latest_version}...")
            fg = self.fs.get_feature_group(name=feature_group_name, version=latest_version)
            df = fg.read()
            print(f"✅ Successfully loaded features from Hopsworks: {len(df)} records, {len(df.columns)} columns")
            
            # Sort by datetime
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # Prepare data similar to main training script
            df = self._preprocess_data(df)
            
            # Create temporal split (70% train, 30% test for optimization)
            split_idx = int(len(df) * 0.7)
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
            
            # Prepare features and target
            target_col = 'aqi_epa_calc'
            feature_cols = [col for col in df.columns if col not in ['datetime', target_col]]
            
            self.X_train = train_df[feature_cols]
            self.X_test = test_df[feature_cols]
            self.y_train = train_df[target_col].values
            self.y_test = test_df[target_col].values
            self.feature_cols = feature_cols
            
            print(f"✅ Data prepared - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
            
            # Handle missing values
            self._handle_missing_values()
            
            # Scale features
            self._scale_features()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def _get_latest_feature_group_version(self, feature_group_name: str) -> int:
        """Get the latest version of a feature group"""
        try:
            feature_groups = self.fs.get_feature_groups(name=feature_group_name)
            if not feature_groups:
                return 0
            latest_version = max([fg.version for fg in feature_groups])
            return latest_version
        except Exception as e:
            print(f"⚠️ Error getting feature group versions: {str(e)}")
            return 0
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic preprocessing similar to main script"""
        print("🔄 Preprocessing data...")
        
        # Handle missing values with forward fill and interpolation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'datetime':
                # Forward fill first
                df[col] = df[col].fillna(method='ffill')
                # Then interpolate
                df[col] = df[col].interpolate(method='linear')
                # Finally, fill any remaining with median
                df[col] = df[col].fillna(df[col].median())
        
        # Create basic lag features
        lag_columns = ['aqi_epa_calc', 'pm2_5_nowcast', 'no2_ppb', 'pm10_nowcast']
        for col in lag_columns:
            if col in df.columns:
                df[f'{col}_lag_1h'] = df[col].shift(1)
        
        # Drop rows with NaN values after lag creation
        df = df.dropna().reset_index(drop=True)
        
        return df
    
    def _handle_missing_values(self):
        """Smart NaN handling with three-tier strategy"""
        print("🔧 Handling missing values...")
        
        # Forward fill (limit 3)
        self.X_train = self.X_train.fillna(method='ffill', limit=3)
        self.X_test = self.X_test.fillna(method='ffill', limit=3)
        
        # Linear interpolation (limit 12)
        self.X_train = self.X_train.interpolate(method='linear', limit=12)
        self.X_test = self.X_test.interpolate(method='linear', limit=12)
        
        # Fill remaining with median
        for col in self.X_train.columns:
            if self.X_train[col].isnull().any():
                median_val = self.X_train[col].median()
                self.X_train[col].fillna(median_val, inplace=True)
                self.X_test[col].fillna(median_val, inplace=True)
    
    def _scale_features(self):
        """Scale features using StandardScaler"""
        print("📏 Scaling features...")
        
        # Identify features to scale (exclude categorical if any)
        features_to_scale = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # Fit scaler on training data
        self.X_train[features_to_scale] = self.scaler.fit_transform(self.X_train[features_to_scale])
        self.X_test[features_to_scale] = self.scaler.transform(self.X_test[features_to_scale])
        
        print(f"✅ Scaled {len(features_to_scale)} features")
    
    def objective(self, trial):
        """Optuna objective function for hyperparameter optimization"""
        
        # Suggest hyperparameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42,
            
            # Core parameters to optimize
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        rmse_scores = []
        r2_scores = []
        mape_scores = []
        
        for train_idx, val_idx in tscv.split(self.X_train):
            X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_fold_train, y_fold_val = self.y_train[train_idx], self.y_train[val_idx]
            
            # Train model
            model = LGBMRegressor(**params)
            model.fit(X_fold_train, y_fold_train)
            
            # Predict
            y_pred = model.predict(X_fold_val)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            r2 = r2_score(y_fold_val, y_pred)
            mape = np.mean(np.abs((y_fold_val - y_pred) / y_fold_val)) * 100
            
            rmse_scores.append(rmse)
            r2_scores.append(r2)
            mape_scores.append(mape)
        
        # Multi-objective: minimize RMSE and MAPE, maximize R²
        avg_rmse = np.mean(rmse_scores)
        avg_r2 = np.mean(r2_scores)
        avg_mape = np.mean(mape_scores)
        
        # Combined objective (lower is better)
        # Normalize metrics and combine
        objective_value = avg_rmse + (1 - avg_r2) * 100 + avg_mape * 0.1
        
        # Report intermediate values for pruning
        trial.report(objective_value, step=0)
        
        return objective_value
    
    def optimize_hyperparameters(self, n_trials=100):
        """Run hyperparameter optimization"""
        print(f"🚀 Starting hyperparameter optimization with {n_trials} trials...")
        
        # Create study
        self.study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        # Optimize
        self.study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best parameters
        self.best_params = self.study.best_params
        
        print(f"✅ Optimization completed!")
        print(f"🏆 Best objective value: {self.study.best_value:.4f}")
        print(f"🔧 Best parameters: {self.best_params}")
        
        return self.best_params
    
    def evaluate_best_model(self):
        """Evaluate the best model on test set"""
        print("📊 Evaluating best model on test set...")
        
        # Add fixed parameters
        final_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42,
            **self.best_params
        }
        
        # Train final model
        best_model = LGBMRegressor(**final_params)
        best_model.fit(self.X_train, self.y_train)
        
        # Predict on test set
        y_pred = best_model.predict(self.X_test)
        
        # Calculate comprehensive metrics
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        
        # Directional accuracy
        y_test_diff = np.diff(self.y_test)
        y_pred_diff = np.diff(y_pred)
        da = np.mean(np.sign(y_test_diff) == np.sign(y_pred_diff))
        
        results = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'directional_accuracy': da,
            'best_params': self.best_params
        }
        
        print(f"🎯 Optimized Model Performance:")
        print(f"   RMSE: {rmse:.3f}")
        print(f"   MAE: {mae:.3f}")
        print(f"   R²: {r2:.3f}")
        print(f"   MAPE: {mape:.1f}%")
        print(f"   DA: {da:.3f}")
        
        return results, best_model
    
    def save_results(self, results):
        """Save optimization results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optuna_optimization_results_{timestamp}.json"
        
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, (np.integer, np.floating)):
                json_results[key] = float(value)
            else:
                json_results[key] = value
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Results saved to {filename}")
        return filename
    
    def plot_optimization_history(self):
        """Plot optimization history and parameter importance"""
        print("📈 Creating optimization plots...")
        
        try:
            # Create simple plots manually
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot objective values over trials
            trials = self.study.trials
            values = [trial.value for trial in trials if trial.value is not None]
            ax1.plot(values, 'b-', alpha=0.7)
            ax1.set_xlabel('Trial')
            ax1.set_ylabel('Objective Value')
            ax1.set_title('Optimization Progress')
            ax1.grid(True, alpha=0.3)
            
            # Plot best value over time
            best_values = []
            best_so_far = float('inf')
            for value in values:
                if value < best_so_far:
                    best_so_far = value
                best_values.append(best_so_far)
            
            ax2.plot(best_values, 'r-', linewidth=2)
            ax2.set_xlabel('Trial')
            ax2.set_ylabel('Best Objective Value')
            ax2.set_title('Best Value Progress')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('optimization_summary.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print("✅ Optimization plots saved!")
            
        except Exception as e:
            print(f"⚠️ Error creating plots: {e}")

def main():
    """Main optimization workflow"""
    print("🔧 AQI Hyperparameter Optimization Starting...")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = AQIHyperparameterOptimizer()
    
    # Connect to Hopsworks
    if not optimizer.connect_to_hopsworks():
        return
    
    # Load and prepare data
    if not optimizer.load_and_prepare_data():
        return
    
    # Run optimization
    best_params = optimizer.optimize_hyperparameters(n_trials=50)  # Reduced for faster execution
    
    # Evaluate best model
    results, best_model = optimizer.evaluate_best_model()
    
    # Save results
    results_file = optimizer.save_results(results)
    
    # Plot optimization history
    optimizer.plot_optimization_history()
    
    print("\n" + "=" * 60)
    print("🎉 Hyperparameter Optimization Complete!")
    print(f"📁 Results saved to: {results_file}")
    print("🔧 Use the best_params in your main training script for improved performance")
    
    return results, best_model

if __name__ == "__main__":
    main()