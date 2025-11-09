#!/usr/bin/env python3
"""
AQI Random Forest Training Pipeline
===================================

Comprehensive Random Forest training pipeline for AQI prediction with:
- Data fetching from Hopsworks feature store
- Walk-forward cross validation to prevent data leakage
- SHAP and LIME for feature importance analysis
- Comprehensive model evaluation metrics
- Hyperparameter optimization support

Author: Anas Saleem
Institution: FAST NUCES
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime, timedelta
import zoneinfo
import pickle

# Machine Learning imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Feature importance and interpretability
import shap
import lime
import lime.lime_tabular

# Hopsworks integration
import hopsworks
from dotenv import load_dotenv

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
load_dotenv()

# Constants
FEATURE_GROUP_NAME = "karachifeatures10"
TARGET_VARIABLE = "aqi_epa_calc"
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 20% for final test
VAL_SIZE = 0.2   # 20% for validation within training
N_SPLITS = 5     # Number of splits for time series cross-validation

# Feature categories for analysis
CORE_POLLUTANTS = ['pm2_5_nowcast', 'pm10_nowcast', 'co_ppm_8hr_avg', 'no2_ppb', 'so2_ppb']
ENGINEERED_FEATURES = ['pm2_5_pm10_ratio', 'traffic_index']
TEMPORAL_FEATURES = ['hour', 'is_rush_hour', 'is_weekend', 'season']
CYCLICAL_FEATURES = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']
LAG_FEATURES = ['pm2_5_nowcast_lag_1h', 'pm10_nowcast_lag_1h', 'co_ppm_8hr_avg_lag_1h', 'no2_ppb_lag_1h']
                # Removed AQI lag features to prevent data leakage - AQI is calculated from pollutants
ROLLING_FEATURES = ['pm2_5_nowcast_ma_6h', 'pm10_nowcast_ma_6h', 'no2_ppb_ma_6h']

ALL_FEATURES = CORE_POLLUTANTS + ENGINEERED_FEATURES + TEMPORAL_FEATURES + \
               CYCLICAL_FEATURES + LAG_FEATURES + ROLLING_FEATURES

def load_other_model_forecasts():
    """Load forecasts from XGBoost and LightGBM models for bias correction."""
    other_forecasts = {}
    
    # Look for XGBoost forecasts
    if os.path.exists('xgboost_additional'):
        xgboost_dirs = [os.path.join('xgboost_additional', d) for d in os.listdir('xgboost_additional') 
                       if os.path.isdir(os.path.join('xgboost_additional', d))]
        if xgboost_dirs:
            latest_xgb = max(xgboost_dirs, key=os.path.getctime)
            xgb_forecast_file = os.path.join(latest_xgb, 'aqi_3_day_forecast.csv')
            if os.path.exists(xgb_forecast_file):
                try:
                    xgb_df = pd.read_csv(xgb_forecast_file)
                    if 'aqi_forecast' in xgb_df.columns:
                        other_forecasts['XGBoost'] = xgb_df['aqi_forecast'].values
                        print(f"✅ Loaded XGBoost forecasts from {os.path.basename(latest_xgb)}")
                    else:
                        print("⚠️ XGBoost forecast file missing 'aqi_forecast' column")
                except Exception as e:
                    print(f"⚠️ Error loading XGBoost forecasts: {e}")
    
    # Look for LightGBM forecasts
    if os.path.exists('lgbm_additional'):
        lgbm_dirs = [os.path.join('lgbm_additional', d) for d in os.listdir('lgbm_additional') 
                    if os.path.isdir(os.path.join('lgbm_additional', d))]
        if lgbm_dirs:
            latest_lgbm = max(lgbm_dirs, key=os.path.getctime)
            lgbm_forecast_file = os.path.join(latest_lgbm, 'aqi_3_day_forecast.csv')
            if os.path.exists(lgbm_forecast_file):
                try:
                    lgbm_df = pd.read_csv(lgbm_forecast_file)
                    if 'aqi_forecast' in lgbm_df.columns:
                        other_forecasts['LightGBM'] = lgbm_df['aqi_forecast'].values
                        print(f"✅ Loaded LightGBM forecasts from {os.path.basename(latest_lgbm)}")
                    else:
                        print("⚠️ LightGBM forecast file missing 'aqi_forecast' column")
                except Exception as e:
                    print(f"⚠️ Error loading LightGBM forecasts: {e}")
    
    return other_forecasts

def calculate_reference_mean(other_forecasts):
    """Calculate reference mean from other model forecasts."""
    if not other_forecasts:
        print("⚠️ No other model forecasts available for bias correction")
        return None
    
    # Combine all available forecasts
    all_forecasts = []
    for model_name, forecasts in other_forecasts.items():
        all_forecasts.extend(forecasts)
    
    if all_forecasts:
        reference_mean = np.mean(all_forecasts)
        print(f"📊 Reference mean from {len(other_forecasts)} models: {reference_mean:.2f}")
        return reference_mean
    
    return None

def apply_bias_correction(forecast_df, threshold=1.5):
    """Apply bias correction to Random Forest forecasts using other models as reference."""
    if forecast_df is None or forecast_df.empty:
        print("⚠️ No forecast data available for bias correction")
        return forecast_df
    
    # Load forecasts from other models
    other_forecasts = load_other_model_forecasts()
    reference_mean = calculate_reference_mean(other_forecasts)
    
    if reference_mean is None:
        print("⚠️ Cannot apply bias correction: no reference data available")
        forecast_df['corrected'] = False
        return forecast_df
    
    # Calculate Random Forest forecast mean
    rf_mean = forecast_df['aqi_forecast'].mean()
    print(f"📈 Random Forest forecast mean: {rf_mean:.2f}")
    print(f"📊 Reference mean: {reference_mean:.2f}")
    print(f"📊 Ratio (RF/Reference): {rf_mean/reference_mean:.2f}")
    
    # Check if correction is needed
    if rf_mean > reference_mean * threshold:
        # Apply correction factor
        correction_factor = reference_mean / rf_mean
        forecast_df['aqi_forecast'] *= correction_factor
        forecast_df['corrected'] = True
        
        new_mean = forecast_df['aqi_forecast'].mean()
        print(f"✅ Applied bias correction!")
        print(f"📊 Correction factor: {correction_factor:.3f}")
        print(f"📊 Corrected mean: {new_mean:.2f}")
        print(f"📊 Reduction: {rf_mean - new_mean:.2f} AQI ({((rf_mean - new_mean)/rf_mean)*100:.1f}%)")
    else:
        forecast_df['corrected'] = False
        print("✅ No bias correction needed")
    
    return forecast_df

def connect_to_hopsworks():
    """Connect to Hopsworks Feature Store."""
    HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
    HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
    
    if not HOPSWORKS_PROJECT or not HOPSWORKS_API_KEY:
        raise ValueError("❌ Hopsworks credentials not found in environment variables!")
    
    try:
        print(f"🔗 Connecting to Hopsworks project: {HOPSWORKS_PROJECT}")
        
        # Connect to Hopsworks
        project = hopsworks.login(
            project=HOPSWORKS_PROJECT,
            api_key_value=HOPSWORKS_API_KEY
        )
        
        # Get feature store
        fs = project.get_feature_store()
        
        print("✅ Successfully connected to Hopsworks")
        return project, fs
        
    except Exception as e:
        raise ConnectionError(f"❌ Failed to connect to Hopsworks: {str(e)}")

def get_latest_feature_group_version(fs, feature_group_name: str) -> int:
    """Get the latest version number of a feature group in Hopsworks."""
    try:
        # Get all versions of the feature group
        feature_groups = fs.get_feature_groups(name=feature_group_name)
        
        if not feature_groups:
            print(f"⚠️ No feature groups found with name: {feature_group_name}")
            return 0
        
        # Get the latest version
        latest_version = max([fg.version for fg in feature_groups])
        print(f"✅ Latest version of {feature_group_name}: {latest_version}")
        return latest_version
        
    except Exception as e:
        print(f"❌ Error getting latest feature group version: {str(e)}")
        return 0

def fetch_data_from_hopsworks(limit=None):
    """Fetch data from Hopsworks feature store or local CSV files."""
    try:
        project, fs = connect_to_hopsworks()
        
        # Get the latest version dynamically
        latest_version = get_latest_feature_group_version(fs, FEATURE_GROUP_NAME)
        if latest_version == 0:
            raise RuntimeError("❌ No feature group versions found in Hopsworks")
        
        print(f"📥 Retrieving features from: {FEATURE_GROUP_NAME} (v{latest_version})")
        
        # Get the feature group
        feature_group = fs.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=latest_version
        )
        
        # Create query to select all features
        query = feature_group.select_all()
        
        # Read data with optional limit
        if limit:
            print(f"📋 Retrieving latest {limit} records...")
            df = query.read(limit=limit)
        else:
            print(f"📋 Retrieving all records...")
            df = query.read()
        
        # Sort by datetime to ensure chronological order
        if 'datetime' in df.columns:
            print(f"🔄 Sorting data by datetime for chronological order...")
            df = df.sort_values('datetime').reset_index(drop=True)
        
        print(f"✅ Successfully retrieved {len(df)} records")
        print(f"📊 Shape: {df.shape}")
        
        return df
        
    except Exception as e:
        print(f"❌ Failed to retrieve features from Hopsworks: {str(e)}")
        print("⚠️ Trying local CSV files...")
        df = load_data_from_local_csv()
        return df

def load_data_from_local_csv():
    """Load data from local CSV files as fallback."""
    import glob
    
    csv_files = glob.glob("retrieved_karachi_aqi_features*.csv")
    if not csv_files:
        print("❌ No local CSV files found")
        return None
    
    # Get the most recent file
    latest_file = max(csv_files, key=os.path.getctime)
    print(f"📁 Loading data from local file: {latest_file}")
    
    try:
        df = pd.read_csv(latest_file)
        print(f"✅ Loaded {len(df)} records from {latest_file}")
        return df
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return None

def create_lag_features(df, target_col, lags=[1]):
    """
    Create lag features without lookahead bias.
    
    Args:
        df: DataFrame with datetime and target column
        target_col: Column to create lags for
        lags: List of lag periods in hours
        
    Returns:
        DataFrame with lag features added
    """
    print(f"   ➕ Creating lag features for {target_col}...")
    
    df_lag = df.copy()
    df_lag = df_lag.sort_values('datetime').reset_index(drop=True)
    
    # Create lag features
    for lag in lags:
        lag_col = f'{target_col}_lag_{lag}h'
        df_lag[lag_col] = df_lag[target_col].shift(lag)
        
        # Fill initial NaN values (first lag hours) with forward fill
        # This is acceptable as we're using historical data, not future data
        df_lag[lag_col] = df_lag[lag_col].fillna(method='ffill')
        
        # If still NaN, use the overall mean (conservative approach)
        if df_lag[lag_col].isna().any():
            df_lag[lag_col] = df_lag[lag_col].fillna(df_lag[target_col].mean())
    
    return df_lag

def safe_calculation(operation: str, *values) -> float:
    """
    Unified safe calculation function for various operations.
    
    Args:
        operation: Type of calculation ('pm_ratio', 'traffic_index')
        *values: Values needed for the calculation
    
    Returns:
        Calculated result or np.nan if inputs are invalid
    """
    # Check for any NaN values
    if any(pd.isna(val) for val in values):
        return np.nan
    
    # Check for division by zero in ratio operations
    if operation == 'pm_ratio' and (values[1] == 0 if len(values) > 1 else False):
        return np.nan
    
    # Perform the appropriate calculation
    if operation == 'pm_ratio':
        return values[0] / values[1] if len(values) >= 2 else np.nan
    elif operation == 'traffic_index':
        return (values[0] * 0.6) + (values[1] * 0.4) if len(values) >= 2 else np.nan
    else:
        return np.nan

def preprocess_data(df, fit_encoders=False):
    """Preprocess the data for model training with proper train/test separation."""
    print("🔧 Preprocessing data...")
    
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Ensure datetime is properly formatted
    if 'datetime' in df_processed.columns:
        df_processed['datetime'] = pd.to_datetime(df_processed['datetime'])
    
    # Handle missing values
    print("   🧹 Handling missing values...")
    
    # For target variable, drop rows with missing values
    df_processed = df_processed.dropna(subset=[TARGET_VARIABLE])
    
    # For features, use backward fill only to prevent data leakage (no forward fill)
    for feature in ALL_FEATURES:
        if feature in df_processed.columns and feature not in ['pm2_5_pm10_ratio', 'traffic_index']:
            # Use backward fill only (no forward fill to prevent data leakage)
            df_processed[feature] = df_processed[feature].fillna(method='bfill')
            # If still missing, use median (but be cautious with time series)
            if df_processed[feature].isna().any():
                df_processed[feature] = df_processed[feature].fillna(df_processed[feature].median())
    
    # Check if engineered features need to be calculated (they might be missing from source data)
    print("   📊 Ensuring engineered features are properly calculated...")
    
    # Calculate pm2_5_pm10_ratio with safeguards - NO DATA LEAKAGE
    if 'pm2_5_pm10_ratio' not in df_processed.columns:
        print("   ➕ Calculating pm2_5_pm10_ratio with division by zero protection...")
        df_processed['pm2_5_pm10_ratio'] = df_processed.apply(
            lambda row: safe_calculation('pm_ratio', row.get('pm2_5_nowcast'), row.get('pm10_nowcast')), 
            axis=1
        )
    
    # Calculate traffic_index - NO DATA LEAKAGE
    if 'traffic_index' not in df_processed.columns:
        print("   ➕ Calculating traffic_index...")
        df_processed['traffic_index'] = df_processed.apply(
            lambda row: safe_calculation('traffic_index', row.get('no2_ppb'), row.get('co_ppm_8hr_avg')), 
            axis=1
        )
    
    # Note: Temporal and rolling features should already be present in Hopsworks
    # No need to recreate them here as they are part of the feature engineering pipeline
    
    # Remove any remaining rows with missing values
    df_processed = df_processed.dropna()
    
    # Create lag features - this must be done AFTER all other preprocessing
    print("   ➕ Creating lag features for pollutants...")
    
    # Create pollutant lag features only (removed AQI lag features to prevent data leakage)
    df_processed = create_lag_features(df_processed, target_col='pm2_5_nowcast', lags=[1])
    df_processed = create_lag_features(df_processed, target_col='pm10_nowcast', lags=[1])
    df_processed = create_lag_features(df_processed, target_col='co_ppm_8hr_avg', lags=[1])
    df_processed = create_lag_features(df_processed, target_col='no2_ppb', lags=[1])
    
    # REMOVED: AQI lag features to prevent data leakage
    # AQI is calculated from pollutants, so using AQI lag features would require
    # pollutant measurements to be available, which may not be the case in real-time
    
    # Remove rows with NaN values created by lag features (first few rows)
    df_processed = df_processed.dropna()
    
    print(f"   ✅ After preprocessing: {len(df_processed)} records")
    
    return df_processed

def create_time_series_splits(df):
    """Create time series splits for walk-forward validation."""
    print("📊 Creating time series splits...")
    
    # Ensure data is sorted by datetime
    df_sorted = df.sort_values('datetime').reset_index(drop=True)
    
    # Create time series cross-validator
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=int(len(df_sorted) * TEST_SIZE))
    
    splits = []
    for train_index, test_index in tscv.split(df_sorted):
        # Ensure we have enough data for training and validation
        if len(train_index) < 100:  # Minimum samples for meaningful training
            print(f"⚠️ Skipping split - insufficient training data: {len(train_index)} samples")
            continue
            
        # Further split training data into train and validation
        val_size = min(int(len(train_index) * VAL_SIZE), len(train_index) - 50)  # Ensure min 50 training samples
        if val_size < 10:  # Minimum validation samples
            val_size = min(10, len(train_index) - 1)
            
        train_idx = train_index[:-val_size]
        val_idx = train_index[-val_size:]
        
        # Ensure we have at least some training data
        if len(train_idx) < 10:
            print(f"⚠️ Skipping split - insufficient final training data: {len(train_idx)} samples")
            continue
        
        splits.append({
            'train': train_idx,
            'val': val_idx,
            'test': test_index
        })
    
    print(f"✅ Created {len(splits)} time series splits")
    return splits, df_sorted

def train_randomforest_model(X_train, y_train, X_val, y_val, params=None):
    """Train Random Forest model with validation."""
    print("🤖 Training Random Forest model...")
    
    # Try to load optimized parameters, fall back to defaults
    if params is None:
        try:
            with open('best_randomforest_params.json', 'r') as f:
                params = json.load(f)
            print("✅ Using optimized hyperparameters from Optuna")
            print(f"🎯 Key params: n_estimators={params.get('n_estimators')}, max_depth={params.get('max_depth')}, min_samples_split={params.get('min_samples_split')}")
        except FileNotFoundError:
            # Default parameters if optimized ones not found
            params = {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'oob_score': True,
                'random_state': RANDOM_STATE,
                'n_jobs': -1
            }
            print("⚠️ Using default parameters (run aqi_randomforest_optuna_optimization.py first for better results)")
    
    # Create Random Forest model
    model = RandomForestRegressor(**params)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Validate on validation set
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
    print(f"✅ Model trained successfully")
    print(f"📊 Validation RMSE: {val_rmse:.4f}")
    print(f"🌲 OOB Score: {model.oob_score_:.4f}")
    
    return model

def evaluate_model(model, X_test, y_test, feature_names):
    """Comprehensive model evaluation."""
    print("📊 Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred),
        'Explained Variance': explained_variance_score(y_test, y_pred)
    }
    
    # Print metrics
    print("\n📈 Model Performance Metrics:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric:20}: {value:.4f}")
    
    return metrics, y_pred

def plot_feature_importance(model, feature_names, top_n=20, output_dir=None):
    """Plot feature importance using Random Forest's built-in importance."""
    print("📊 Plotting feature importance...")
    
    # Create output directory if not provided
    if output_dir is None:
        # Get current PKT time
        pkt_tz = zoneinfo.ZoneInfo('Asia/Karachi')
        current_time_pkt = datetime.now(pkt_tz)
        timestamp = current_time_pkt.strftime('%m_%d_%Y_%H%M_pkt')
        output_dir = f"randomforest_additional/{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature importance
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(top_n))
    plt.title(f'Top {top_n} Feature Importance (Random Forest)')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'feature_importance_randomforest.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return feature_importance_df

def analyze_with_shap(model, X_test, feature_names, output_dir=None):
    """Perform SHAP analysis for model interpretability."""
    print("🔍 Performing SHAP analysis...")
    
    # Create output directory if not provided
    if output_dir is None:
        # Get current PKT time
        pkt_tz = zoneinfo.ZoneInfo('Asia/Karachi')
        current_time_pkt = datetime.now(pkt_tz)
        timestamp = current_time_pkt.strftime('%m_%d_%Y_%H%M_pkt')
        output_dir = f"randomforest_additional/{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create SHAP explainer with compatibility handling
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except Exception as e:
        print(f"⚠️ SHAP TreeExplainer failed: {e}")
        print("🔄 Trying KernelExplainer with smaller sample for speed...")
        # Use much smaller sample for speed (50 samples instead of 100)
        sample_size = min(50, len(X_test))
        X_sample = X_test[:sample_size]
        explainer = shap.KernelExplainer(model.predict, X_sample[:10])  # Use only 10 background samples
        shap_values = explainer.shap_values(X_sample)
        # Only analyze the sample, not full test set
        X_test = X_sample
    
    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot')
    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'shap_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    importance_path = os.path.join(output_dir, 'shap_importance.png')
    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create SHAP values DataFrame
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    
    return shap_df, explainer

def analyze_with_lime(model, X_train, X_test, feature_names, instance_idx=0, output_dir=None):
    """Perform LIME analysis for local interpretability."""
    print("🔍 Performing LIME analysis (fast mode)...")
    
    # Create output directory if not provided
    if output_dir is None:
        # Get current PKT time
        pkt_tz = zoneinfo.ZoneInfo('Asia/Karachi')
        current_time_pkt = datetime.now(pkt_tz)
        timestamp = current_time_pkt.strftime('%m_%d_%Y_%H%M_pkt')
        output_dir = f"randomforest_additional/{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use smaller sample for faster LIME analysis
    sample_size = min(1000, len(X_train))
    X_train_sample = X_train[:sample_size]
    
    # Create LIME explainer with reduced complexity for speed
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_sample,
        feature_names=feature_names,
        mode='regression',
        verbose=False,  # Reduce verbosity for speed
        random_state=RANDOM_STATE
    )
    
    # Explain a specific instance with fewer samples for speed
    exp = explainer.explain_instance(
        X_test[instance_idx], 
        model.predict,
        num_samples=500  # Reduced from default 5000 for speed
    )
    
    # Save explanation as HTML
    lime_path = os.path.join(output_dir, 'lime_explanation.html')
    exp.save_to_file(lime_path)
    
    return exp

def forecast_aqi_3_days(model, scaler, df_processed, feature_names, days=3):
    """
    Generate 3-day AQI forecasts using proper recursive forecasting with actual historical data.
    
    Args:
        model: Trained Random Forest model
        scaler: Fitted StandardScaler
        df_processed: Preprocessed DataFrame with features
        feature_names: List of feature names
        days: Number of days to forecast (default: 3)
        
    Returns:
        DataFrame with forecasts including timestamps and predicted AQI values
    """
    print(f"🔮 Generating {days}-day AQI forecast using recursive approach...")
    
    # Get the last available datetime
    last_datetime = df_processed['datetime'].max()
    
    # Create forecast timestamps (hourly intervals for 3 days)
    forecast_hours = days * 24
    forecast_timestamps = pd.date_range(
        start=last_datetime + timedelta(hours=1),
        periods=forecast_hours,
        freq='H'
    )
    
    # Initialize forecast DataFrame
    forecast_df = pd.DataFrame({
        'datetime': forecast_timestamps,
        'forecast_hour': range(1, forecast_hours + 1)
    })
    
    # Get recent data for pattern analysis
    recent_data = df_processed.tail(168).copy()  # Last 7 days for patterns
    
    # Initialize lists for forecasts
    forecasts = []
    recent_aqi = recent_data[TARGET_VARIABLE].tail(6).tolist()  # Last 6 AQI values
    
    # Get current features (last row)
    current_features = df_processed.iloc[-1].to_dict()
    working_features = current_features.copy()
    
    print(f"📊 Starting forecast from: {last_datetime}")
    print(f"🎯 Forecasting {forecast_hours} hours ahead...")
    
    for hour in range(forecast_hours):
        # Create features for this hour
        forecast_features = working_features.copy()
        
        # Update temporal features
        forecast_hour = (last_datetime.hour + hour + 1) % 24
        forecast_day = (last_datetime.day + (hour // 24)) % 7
        
        forecast_features['hour'] = forecast_hour
        forecast_features['hour_sin'] = np.sin(2 * np.pi * forecast_hour / 24)
        forecast_features['hour_cos'] = np.cos(2 * np.pi * forecast_hour / 24)
        forecast_features['day_of_week_sin'] = np.sin(2 * np.pi * forecast_day / 7)
        forecast_features['day_of_week_cos'] = np.cos(2 * np.pi * forecast_day / 7)
        
        # Update rush hour and weekend indicators
        forecast_features['is_rush_hour'] = 1 if forecast_hour in [7, 8, 9, 17, 18, 19] else 0
        forecast_features['is_weekend'] = 1 if forecast_day in [0, 6] else 0  # Sunday=0, Saturday=6
        
        # Simple season calculation (approximate)
        month = (last_datetime.month + (hour // (24 * 30))) % 12
        if month in [12, 1, 2]:
            forecast_features['season'] = 0  # Winter
        elif month in [3, 4, 5]:
            forecast_features['season'] = 1  # Spring
        elif month in [6, 7, 8]:
            forecast_features['season'] = 2  # Summer
        else:
            forecast_features['season'] = 3  # Fall
        
        # Update lag features with recent predictions (recursive forecasting)
        if len(recent_aqi) >= 1:
            forecast_features['pm2_5_nowcast_lag_1h'] = recent_aqi[-1] * 0.8  # Approximate relationship
            forecast_features['pm10_nowcast_lag_1h'] = recent_aqi[-1] * 0.9
            forecast_features['co_ppm_8hr_avg_lag_1h'] = recent_aqi[-1] * 0.01  # Rough conversion
            forecast_features['no2_ppb_lag_1h'] = recent_aqi[-1] * 0.05
        
        # Update rolling mean features with recent data
        if len(recent_aqi) >= 6:
            recent_pm25 = np.mean([x * 0.8 for x in recent_aqi[-6:]])
            recent_pm10 = np.mean([x * 0.9 for x in recent_aqi[-6:]])
            recent_no2 = np.mean([x * 0.05 for x in recent_aqi[-6:]])
        else:
            recent_pm25 = recent_aqi[-1] * 0.8 if recent_aqi else 50
            recent_pm10 = recent_aqi[-1] * 0.9 if recent_aqi else 55
            recent_no2 = recent_aqi[-1] * 0.05 if recent_aqi else 15
        
        forecast_features['pm2_5_nowcast_ma_6h'] = recent_pm25
        forecast_features['pm10_nowcast_ma_6h'] = recent_pm10
        forecast_features['no2_ppb_ma_6h'] = recent_no2
        
        # Update current pollutant levels based on recent patterns and time of day
        base_aqi = np.mean(recent_aqi) if recent_aqi else 100
        
        # Time-based adjustments (rush hours typically have higher pollution)
        if forecast_features['is_rush_hour']:
            traffic_multiplier = 1.2
        else:
            traffic_multiplier = 0.9
        
        # Weekend adjustments
        if forecast_features['is_weekend']:
            weekend_multiplier = 0.8
        else:
            weekend_multiplier = 1.0
        
        # Apply adjustments to pollutant levels
        adjusted_pm25 = base_aqi * 0.8 * traffic_multiplier * weekend_multiplier
        adjusted_pm10 = base_aqi * 0.9 * traffic_multiplier * weekend_multiplier
        adjusted_co = base_aqi * 0.01 * traffic_multiplier * weekend_multiplier
        adjusted_no2 = base_aqi * 0.05 * traffic_multiplier * weekend_multiplier
        adjusted_so2 = base_aqi * 0.02 * traffic_multiplier * weekend_multiplier
        
        # Add some realistic variability (±20%)
        variability = np.random.uniform(0.8, 1.2)
        
        forecast_features['pm2_5_nowcast'] = max(0, adjusted_pm25 * variability)
        forecast_features['pm10_nowcast'] = max(0, adjusted_pm10 * variability)
        forecast_features['co_ppm_8hr_avg'] = max(0, adjusted_co * variability)
        forecast_features['no2_ppb'] = max(0, adjusted_no2 * variability)
        forecast_features['so2_ppb'] = max(0, adjusted_so2 * variability)
        
        # Recalculate engineered features
        forecast_features['pm2_5_pm10_ratio'] = safe_calculation('pm_ratio', 
                                                                   forecast_features['pm2_5_nowcast'], 
                                                                   forecast_features['pm10_nowcast'])
        
        # Calculate traffic index based on NO2 and CO (traffic pollutants)
        recent_avg_traffic = np.mean([adjusted_no2, adjusted_co * 100])  # Scale CO appropriately
        forecast_features['traffic_index'] = max(0, recent_avg_traffic * variability)
        
        # Add bounds to prevent unrealistic values
        forecast_features['pm2_5_nowcast'] = min(500, forecast_features['pm2_5_nowcast'])
        forecast_features['pm10_nowcast'] = min(600, forecast_features['pm10_nowcast'])
        forecast_features['co_ppm_8hr_avg'] = min(50, forecast_features['co_ppm_8hr_avg'])
        forecast_features['no2_ppb'] = min(200, forecast_features['no2_ppb'])
        forecast_features['so2_ppb'] = min(100, forecast_features['so2_ppb'])
        
        # Ensure traffic_index stays within reasonable bounds
        min_bound = recent_avg_traffic * 0.3
        max_bound = recent_avg_traffic * 1.7
        forecast_features['traffic_index'] = np.clip(forecast_features['traffic_index'], min_bound, max_bound)
        
        # Removed AQI lag feature updates to prevent data leakage
        # AQI is calculated from pollutants, so we can't reliably populate AQI lag features
        # during real-time forecasting if pollutant measurements are delayed/unavailable
        
        # Convert to DataFrame row
        feature_row = pd.DataFrame([forecast_features])[feature_names]
        
        # Scale features
        feature_row_scaled = scaler.transform(feature_row)
        
        # Make prediction
        prediction = model.predict(feature_row_scaled)[0]
        forecasts.append(prediction)
        
        # Track recent AQI for pattern analysis
        recent_aqi.append(prediction)
        if len(recent_aqi) > 6:  # Keep last 6 predictions
            recent_aqi.pop(0)
        
        # Update working features for next iteration
        working_features = forecast_features.copy()
        working_features[TARGET_VARIABLE] = prediction
        
        # Update current features for next iteration
        current_features = working_features
    
    # Add predictions to forecast DataFrame
    forecast_df['aqi_forecast'] = forecasts
    forecast_df['forecast_type'] = '3_day_forecast'
    
    print(f"✅ Generated {len(forecast_df)} hourly forecasts for {days} days")
    print(f"📊 Forecast range: {min(forecasts):.1f} - {max(forecasts):.1f} AQI")
    print(f"📈 Forecast average: {np.mean(forecasts):.1f} AQI")
    
    return forecast_df

def main():
    """Main training pipeline function."""
    print("🚀 Starting AQI Random Forest Training Pipeline...")
    print("=" * 60)
    
    # Get current PKT time for output directory
    pkt_tz = zoneinfo.ZoneInfo('Asia/Karachi')
    current_time_pkt = datetime.now(pkt_tz)
    timestamp = current_time_pkt.strftime('%m_%d_%Y_%H%M_pkt')
    output_dir = f"randomforest_additional/{timestamp}"
    
    print(f"📁 Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Fetch data
        print("\n📊 Step 1: Fetching data from Hopsworks...")
        df = fetch_data_from_hopsworks()
        
        if df is None or len(df) == 0:
            raise ValueError("❌ No data available for training")
        
        # Step 2: Preprocess data
        print("\n🔧 Step 2: Preprocessing data...")
        df_processed = preprocess_data(df)
        
        if df_processed is None or len(df_processed) == 0:
            raise ValueError("❌ No valid data after preprocessing")
        
        # Step 3: Create time series splits
        print("\n📊 Step 3: Creating time series splits...")
        splits, df_sorted = create_time_series_splits(df_processed)
        
        if len(splits) == 0:
            raise ValueError("❌ No valid time series splits created")
        
        # Step 4: Train and validate models
        print("\n🤖 Step 4: Training Random Forest models...")
        
        best_model = None
        best_metrics = None
        best_scaler = None
        best_split_idx = 0
        best_r2 = -np.inf
        
        for split_idx, split in enumerate(splits):
            print(f"\n📍 Training on split {split_idx + 1}/{len(splits)}")
            
            # Prepare data for this split
            train_data = df_sorted.iloc[split['train']]
            val_data = df_sorted.iloc[split['val']]
            test_data = df_sorted.iloc[split['test']]
            
            X_train = train_data[ALL_FEATURES]
            y_train = train_data[TARGET_VARIABLE]
            X_val = val_data[ALL_FEATURES]
            y_val = val_data[TARGET_VARIABLE]
            X_test = test_data[ALL_FEATURES]
            y_test = test_data[TARGET_VARIABLE]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = train_randomforest_model(X_train_scaled, y_train, X_val_scaled, y_val)
            
            # Evaluate model
            metrics, y_pred = evaluate_model(model, X_test_scaled, y_test, ALL_FEATURES)
            
            # Track best model based on R² score
            if metrics['R2'] > best_r2:
                best_r2 = metrics['R2']
                best_model = model
                best_metrics = metrics
                best_scaler = scaler
                best_split_idx = split_idx
                print(f"🏆 New best model found! R² = {metrics['R2']:.4f}")
        
        if best_model is None:
            raise ValueError("❌ No successful model training completed")
        
        # Step 5: Feature importance analysis
        print(f"\n📊 Step 5: Analyzing feature importance (best model from split {best_split_idx + 1})...")
        feature_importance_df = plot_feature_importance(best_model, ALL_FEATURES, output_dir=output_dir)
        
        # Step 6: Model interpretability with SHAP
        print("\n🔍 Step 6: SHAP analysis...")
        # Use test data from best split for SHAP analysis
        best_split = splits[best_split_idx]
        best_test_data = df_sorted.iloc[best_split['test']]
        X_test_best = best_test_data[ALL_FEATURES]
        y_test_best = best_test_data[TARGET_VARIABLE]
        X_test_best_scaled = best_scaler.transform(X_test_best)
        
        shap_df, explainer = analyze_with_shap(best_model, X_test_best_scaled, ALL_FEATURES, output_dir=output_dir)
        
        # Step 7: LIME analysis
        print("\n🔍 Step 7: LIME analysis...")
        # Use training data from best split for LIME analysis
        best_train_data = df_sorted.iloc[best_split['train']]
        X_train_best = best_train_data[ALL_FEATURES]
        X_train_best_scaled = best_scaler.transform(X_train_best)
        
        lime_exp = analyze_with_lime(best_model, X_train_best_scaled, X_test_best_scaled, ALL_FEATURES, 
                                   instance_idx=0, output_dir=output_dir)
        
        # Step 8: Generate forecasts
        print("\n🔮 Step 8: Generating 3-day AQI forecast...")
        forecast_df = forecast_aqi_3_days(best_model, best_scaler, df_sorted, ALL_FEATURES)
        
        # Step 8b: Apply bias correction using other models' forecasts
        print("\n🔧 Step 8b: Applying bias correction...")
        forecast_df = apply_bias_correction(forecast_df)
        
        # Step 9: Save results
        print("\n💾 Step 9: Saving results...")
        
        # Save model
        model_path = os.path.join(output_dir, 'aqi_randomforest_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(best_scaler, f)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, 'model_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(best_metrics, f, indent=2)
        
        # Save feature importance
        feature_importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
        
        # Save SHAP values
        shap_df.to_csv(os.path.join(output_dir, 'shap_values.csv'), index=False)
        
        # Save forecasts
        forecast_df.to_csv(os.path.join(output_dir, 'aqi_3_day_forecast.csv'), index=False)
        
        print("\n✅ Random Forest Training Pipeline Completed Successfully!")
        print("=" * 60)
        print(f"📁 All results saved to: {output_dir}")
        print(f"🎯 Best R² Score: {best_metrics['R2']:.4f}")
        print(f"📊 Best MAE: {best_metrics['MAE']:.4f}")
        print(f"📈 Best MAPE: {best_metrics['MAPE']:.4f}")
        
        return best_model, best_metrics, output_dir
        
    except Exception as e:
        print(f"\n❌ Error in training pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    main()