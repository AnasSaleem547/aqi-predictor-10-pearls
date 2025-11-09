#!/usr/bin/env python3
"""
XGBoost Hyperparameter Optimization using Optuna for AQI Prediction
This script performs hyperparameter tuning for XGBoost model using Optuna.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import zoneinfo
import hopsworks
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2
N_TRIALS = 50  # Number of Optuna trials
TIMEOUT = 3600  # 1 hour timeout

# Hopsworks configuration
HOPSWORKS_API_KEY = os.environ.get('HOPSWORKS_API_KEY')
HOPSWORKS_PROJECT_NAME = os.environ.get('HOPSWORKS_PROJECT', 'airquality')  
FEATURE_GROUP_NAME = 'karachifeatures10'
FEATURE_GROUP_VERSION = 1

# Feature configuration
PRODUCTION_FEATURES = [
    # Pollutant concentrations
    'pm2_5_nowcast', 'pm10_nowcast', 'co_ppm_8hr_avg', 
    'no2_ppb', 'so2_ppb', 'o3_ppb_8hr_avg',
    
    # Weather features  
    'temperature_f', 'humidity_percent', 'wind_speed_mph', 'pressure_hpa',
    'visibility_miles', 'uv_index', 'cloud_cover_percent',
    
    # Traffic and location
    'traffic_index', 'distance_to_industrial_km', 'distance_to_main_road_km',
    
    # Temporal features
    'hour', 'is_rush_hour', 'is_weekend', 'season',
    
    # Cyclical features
    'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
    
    # Lag features (pollutants only - NO AQI lag features to prevent data leakage)
    'pm2_5_nowcast_lag_1h', 'pm10_nowcast_lag_1h', 'co_ppm_8hr_avg_lag_1h', 'no2_ppb_lag_1h',
    'pm2_5_nowcast_lag_2h', 'pm10_nowcast_lag_2h', 'co_ppm_8hr_avg_lag_2h', 'no2_ppb_lag_2h',
    'pm2_5_nowcast_lag_3h', 'pm10_nowcast_lag_3h', 'co_ppm_8hr_avg_lag_3h', 'no2_ppb_lag_3h',
    
    # Rolling mean features (pollutants only)
    'pm2_5_nowcast_ma_6h', 'pm10_nowcast_ma_6h', 'no2_ppb_ma_6h',
    'pm2_5_nowcast_ma_12h', 'pm10_nowcast_ma_12h', 'no2_ppb_ma_12h',
    
    # Engineered features
    'pm2_5_pm10_ratio', 'temp_humidity_interaction',
    'wind_pollution_interaction', 'pressure_pollution_interaction'
]

def fetch_data_from_hopsworks():
    """Fetch data from Hopsworks feature store."""
    print("🔌 Connecting to Hopsworks feature store...")
    
    try:
        # Connect to Hopsworks
        if HOPSWORKS_API_KEY:
            project = hopsworks.login(
                api_key_value=HOPSWORKS_API_KEY,
                project=HOPSWORKS_PROJECT_NAME
            )
        else:
            project = hopsworks.login(project=HOPSWORKS_PROJECT_NAME)
        
        fs = project.get_feature_store()
        
        # Get feature group
        feature_group = fs.get_feature_group(
            name=FEATURE_GROUP_NAME, 
            version=FEATURE_GROUP_VERSION
        )
        
        # Read all data
        query = feature_group.select_all()
        df = query.read()
        
        print(f"✅ Successfully retrieved {len(df)} records from Hopsworks")
        return df
        
    except Exception as e:
        print(f"❌ Hopsworks connection failed: {e}")
        print("🔄 Falling back to local CSV file...")
        
        # Fallback to local CSV
        try:
            csv_files = [f for f in os.listdir('.') if f.startswith('retrieved_karachi_aqi_features') and f.endswith('.csv')]
            if csv_files:
                latest_file = max(csv_files, key=os.path.getctime)
                print(f"📁 Loading from local file: {latest_file}")
                df = pd.read_csv(latest_file)
                
                # Convert datetime column if it exists
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                
                print(f"✅ Successfully loaded {len(df)} records from local CSV")
                return df
            else:
                raise FileNotFoundError("No local CSV files found")
                
        except FileNotFoundError:
            print("❌ No local data files available")
            return None

def preprocess_data(df):
    """Preprocess the data for XGBoost training."""
    print("🔧 Preprocessing data...")
    
    # Sort by datetime
    df_sorted = df.sort_values('datetime').reset_index(drop=True)
    
    # Handle missing values
    df_clean = df_sorted.copy()
    
    # Forward fill for missing values (but not backward fill to avoid data leakage)
    df_clean = df_clean.fillna(method='ffill')
    
    # Drop remaining NaN values
    initial_len = len(df_clean)
    df_clean = df_clean.dropna()
    final_len = len(df_clean)
    
    print(f"   📊 Dropped {initial_len - final_len} rows with missing values")
    print(f"   📊 Final dataset: {final_len} records")
    
    return df_clean

def create_time_series_splits(df, n_splits=5):
    """Create time series splits for cross-validation."""
    print("📊 Creating time series splits...")
    
    # Sort by datetime
    df_sorted = df.sort_values('datetime').reset_index(drop=True)
    
    # Create splits
    splits = []
    total_size = len(df_sorted)
    split_size = total_size // (n_splits + 1)
    
    for i in range(n_splits):
        train_end = (i + 1) * split_size
        val_start = train_end
        val_end = min(val_start + split_size // 4, total_size)  # Smaller validation set
        
        if val_end - val_start < 100:  # Skip if validation set is too small
            continue
            
        train_idx = list(range(0, train_end))
        val_idx = list(range(val_start, val_end))
        
        splits.append({
            'train': train_idx,
            'val': val_idx
        })
    
    print(f"✅ Created {len(splits)} time series splits")
    return splits, df_sorted

def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for XGBoost hyperparameter optimization."""
    
    # Suggest hyperparameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': 'gbtree',
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    # Create and train model
    model = xgb.XGBRegressor(**params)
    
    # Train with early stopping (use smaller number of rounds for speed)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predict and evaluate
    y_pred = model.predict(X_val)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    return rmse

def run_optimization():
    """Run the complete hyperparameter optimization pipeline."""
    print("🚀 Starting XGBoost Hyperparameter Optimization with Optuna...")
    print("=" * 70)
    
    try:
        # Step 1: Fetch data
        print("\n📊 Step 1: Fetching data...")
        df = fetch_data_from_hopsworks()
        
        if df is None or len(df) == 0:
            raise ValueError("❌ No data available for optimization")
        
        # Step 2: Preprocess data
        print("\n🔧 Step 2: Preprocessing data...")
        df_processed = preprocess_data(df)
        
        if df_processed is None or len(df_processed) == 0:
            raise ValueError("❌ No valid data after preprocessing")
        
        # Step 3: Create features and target
        print("\n🎯 Step 3: Preparing features and target...")
        
        # Ensure all required features exist
        available_features = [f for f in PRODUCTION_FEATURES if f in df_processed.columns]
        missing_features = [f for f in PRODUCTION_FEATURES if f not in df_processed.columns]
        
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
            print(f"✅ Using available features: {available_features}")
        
        X = df_processed[available_features]
        y = df_processed['aqi_epa_calc']
        
        print(f"📊 Feature matrix shape: {X.shape}")
        print(f"📊 Target vector shape: {y.shape}")
        
        # Step 4: Create time series splits
        print("\n📊 Step 4: Creating time series splits...")
        splits, df_sorted = create_time_series_splits(df_processed, n_splits=3)
        
        if len(splits) == 0:
            raise ValueError("❌ No valid time series splits created")
        
        # Step 5: Scale features
        print("\n📏 Step 5: Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use the first split for optimization (to save time)
        split = splits[0]
        X_train = X_scaled[split['train']]
        y_train = y.iloc[split['train']]
        X_val = X_scaled[split['val']]
        y_val = y.iloc[split['val']]
        
        print(f"📊 Training set: {X_train.shape[0]} samples")
        print(f"📊 Validation set: {X_val.shape[0]} samples")
        
        # Step 6: Run Optuna optimization
        print("\n🔬 Step 6: Running Optuna optimization...")
        print(f"🎯 Optimizing for {N_TRIALS} trials with {TIMEOUT}s timeout...")
        
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
        
        # Optimize with progress bar
        with tqdm(total=N_TRIALS, desc="Optuna Trials") as pbar:
            def callback(study, trial):
                pbar.update(1)
                pbar.set_postfix({'Best RMSE': f"{study.best_value:.4f}"})
            
            study.optimize(
                lambda trial: objective(trial, X_train, y_train, X_val, y_val),
                n_trials=N_TRIALS,
                timeout=TIMEOUT,
                callbacks=[callback],
                show_progress_bar=False
            )
        
        # Step 7: Save best parameters
        print("\n💾 Step 7: Saving best parameters...")
        best_params = study.best_params
        
        # Add fixed parameters
        best_params.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        })
        
        # Save to JSON
        with open('best_xgboost_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        
        print(f"✅ Best parameters saved to 'best_xgboost_params.json'")
        print(f"🎯 Best RMSE: {study.best_value:.4f}")
        
        # Step 8: Display optimization results
        print("\n📈 Step 8: Optimization Results...")
        print("=" * 50)
        print("Best Hyperparameters:")
        for param, value in best_params.items():
            if param not in ['objective', 'eval_metric', 'booster', 'random_state', 'n_jobs']:
                print(f"  {param:20}: {value}")
        
        # Step 9: Create optimization visualizations
        print("\n📊 Step 9: Creating optimization visualizations...")
        try:
            # Parameter importance plot
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html('optuna_xgboost_param_importances.html')
            print("✅ Parameter importance plot saved")
            
            # Optimization history plot
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_html('optuna_xgboost_optimization_history.html')
            print("✅ Optimization history plot saved")
            
        except Exception as e:
            print(f"⚠️ Could not create visualizations: {e}")
        
        print("\n🎉 XGBoost hyperparameter optimization completed successfully!")
        print("=" * 70)
        
        return best_params
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run optimization
    best_params = run_optimization()
    
    if best_params:
        print(f"\n✅ Optimization completed! Best RMSE: {best_params.get('best_value', 'N/A')}")
        print("🚀 You can now run the XGBoost training pipeline with optimized parameters!")
    else:
        print("\n❌ Optimization failed. Check the error messages above.")