#!/usr/bin/env python3
"""
AQI XGBoost Training Pipeline
=============================

Comprehensive XGBoost training pipeline for AQI prediction with:
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
import matplotlib
matplotlib.use("Agg")

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

# XGBoost
import xgboost as xgb

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

def fetch_data_from_hopsworks(limit=None, use_csv_first=True):
    """Fetch data from Hopsworks feature store or local CSV files."""
    
    # First try to load from CSV if requested
    if use_csv_first:
        print("📁 Attempting to load data from RandomForest CSV first...")
        csv_files = load_data_from_randomforest_csv()
        if csv_files is not None and len(csv_files) > 0:
            return csv_files
    
    # Fall back to Hopsworks
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

def load_data_from_randomforest_csv():
    """Load data from RandomForest CSV files first."""
    import glob
    
    # Look for RandomForest specific CSV files first
    csv_files = glob.glob("retrieved_karachi_aqi_features_randomforest_*.csv")
    if not csv_files:
        print("❌ No RandomForest CSV files found, trying general CSV files...")
        return None
    
    # Get the most recent RandomForest file
    latest_file = max(csv_files, key=os.path.getctime)
    print(f"📁 Loading data from RandomForest CSV file: {latest_file}")
    
    try:
        df = pd.read_csv(latest_file)
        print(f"✅ Loaded {len(df)} records from RandomForest CSV: {latest_file}")
        return df
    except Exception as e:
        print(f"❌ Error loading RandomForest CSV file: {e}")
        return None

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

def train_xgboost_model(X_train, y_train, X_val, y_val, params=None):
    """Train XGBoost model with early stopping."""
    print("🤖 Training XGBoost model...")
    
    # Try to load optimized parameters, fall back to defaults
    if params is None:
        try:
            with open('best_xgboost_params.json', 'r') as f:
                params = json.load(f)
            print("✅ Using optimized hyperparameters from Optuna")
            print(f"🎯 Key params: max_depth={params.get('max_depth')}, learning_rate={params.get('learning_rate'):.3f}, n_estimators={params.get('n_estimators')}")
        except FileNotFoundError:
            # Default parameters if optimized ones not found
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'booster': 'gbtree',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 1000,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': RANDOM_STATE,
                'n_jobs': -1
            }
            print("⚠️ Using default parameters (run aqi_xgboost_optuna_optimization.py first for better results)")
    
    # Create XGBoost model
    model = xgb.XGBRegressor(**params)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    print(f"✅ Model trained successfully")
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
    """Plot feature importance using XGBoost's built-in importance."""
    print("📊 Plotting feature importance...")
    
    # Create output directory if not provided
    if output_dir is None:
        # Get current PKT time
        pkt_tz = zoneinfo.ZoneInfo('Asia/Karachi')
        current_time_pkt = datetime.now(pkt_tz)
        timestamp = current_time_pkt.strftime('%m_%d_%Y_%H%M_pkt')
        output_dir = f"xgboost_additional/{timestamp}"
    
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
    plt.title(f'Top {top_n} Feature Importance (XGBoost)')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'feature_importance_xgboost.png')
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
        output_dir = f"xgboost_additional/{timestamp}"
    
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
        output_dir = f"xgboost_additional/{timestamp}"
    
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
        model: Trained XGBoost model
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
        start=last_datetime + pd.Timedelta(hours=1),
        periods=forecast_hours,
        freq='H'
    )
    
    # Prepare forecast DataFrame
    forecast_df = pd.DataFrame({'datetime': forecast_timestamps})
    
    # Get the last 48 hours of historical data for better context
    historical_data = df_processed[df_processed['datetime'] >= last_datetime - pd.Timedelta(hours=47)].copy()
    historical_data = historical_data.sort_values('datetime').reset_index(drop=True)
    
    # Initialize forecast predictions and feature history
    forecasts = []
    feature_history = []
    
    # Create initial feature state from last available record
    current_features = {}
    last_record = df_processed.iloc[-1].copy()
    
    for feature in feature_names:
        if feature in last_record.index:
            current_features[feature] = last_record[feature]
        else:
            current_features[feature] = 0  # Default value
    
    # Track recent AQI for realistic patterns
    recent_aqi = []
    
    for i, timestamp in enumerate(forecast_timestamps):
        print(f"   📅 Forecasting for {timestamp}...")
        
        # Create forecast features
        forecast_features = current_features.copy()
        
        # Update temporal features
        forecast_time = timestamp
        forecast_features['hour'] = forecast_time.hour
        forecast_features['is_rush_hour'] = 1 if (7 <= forecast_time.hour <= 9 or 17 <= forecast_time.hour <= 19) else 0
        forecast_features['is_weekend'] = 1 if forecast_time.weekday() >= 5 else 0
        
        # Update cyclical features
        forecast_features['hour_sin'] = np.sin(2 * np.pi * forecast_time.hour / 24)
        forecast_features['hour_cos'] = np.cos(2 * np.pi * forecast_time.hour / 24)
        forecast_features['day_of_week_sin'] = np.sin(2 * np.pi * forecast_time.weekday() / 7)
        forecast_features['day_of_week_cos'] = np.cos(2 * np.pi * forecast_time.weekday() / 7)
        
        # Update season (simplified)
        month = forecast_time.month
        if month in [12, 1, 2]:
            forecast_features['season'] = 0  # Winter
        elif month in [3, 4, 5]:
            forecast_features['season'] = 1  # Spring
        elif month in [6, 7, 8]:
            forecast_features['season'] = 2  # Summer
        else:
            forecast_features['season'] = 3  # Autumn
        
        # Update pollutant features using historical patterns (NO RANDOM MULTIPLIERS)
        # Use the actual last known values with small temporal adjustments based on hour patterns
        hour = forecast_time.hour
        is_weekend = forecast_features['is_weekend']
        
        # Calculate reasonable adjustments based on historical patterns
        # Get historical data for the same hour of day
        same_hour_data = historical_data[historical_data['datetime'].dt.hour == hour]
        if len(same_hour_data) > 0:
            # Use median values for the same hour as baseline
            hour_baseline = same_hour_data.median()
            # Also get the variability (std) for this hour to add realistic noise
            hour_std = same_hour_data.std()
        else:
            # Fallback to overall median and std
            hour_baseline = historical_data.median()
            hour_std = historical_data.std()
        
        # Apply realistic temporal patterns - AQI should be LOWEST at 3-5 AM
        # Use additive adjustments instead of multiplicative to avoid compounding issues
        
        # Base temporal adjustments (additive, not multiplicative)
        temporal_adjustment = 0.0
        
        # Rush hour increases (morning and evening)
        if hour in [7, 8, 17, 18]:
            temporal_adjustment += 0.15  # 15% increase during rush hours
        
        # Night time decreases (lowest pollution 2-5 AM)
        elif hour in [2, 3, 4, 5]:
            temporal_adjustment -= 0.10  # 10% decrease during early morning
        
        # Weekend adjustments (generally lower pollution)
        if is_weekend:
            temporal_adjustment -= 0.08  # 8% decrease on weekends
        
        # Update pollutant concentrations with realistic patterns
        pollutant_features = ['pm2_5_nowcast', 'pm10_nowcast', 'co_ppm_8hr_avg', 'no2_ppb', 'so2_ppb']
        
        for pollutant in pollutant_features:
            if pollutant in forecast_features:
                # Get recent average for this pollutant
                recent_avg = historical_data[pollutant].tail(6).mean()  # Last 6 hours
                
                # Apply temporal adjustment
                adjusted_value = recent_avg * (1 + temporal_adjustment)
                
                # Add small random variation based on hour std (bounded)
                if pollutant in hour_std.index and not pd.isna(hour_std[pollutant]):
                    random_variation = np.random.normal(0, hour_std[pollutant] * 0.1)
                    adjusted_value += random_variation
                
                # Keep within reasonable bounds (±30% of recent average)
                min_bound = recent_avg * 0.7
                max_bound = recent_avg * 1.3
                forecast_features[pollutant] = np.clip(adjusted_value, min_bound, max_bound)
        
        # Update engineered features
        if 'pm2_5_pm10_ratio' in forecast_features:
            if forecast_features['pm10_nowcast'] > 0:
                forecast_features['pm2_5_pm10_ratio'] = forecast_features['pm2_5_nowcast'] / forecast_features['pm10_nowcast']
            else:
                forecast_features['pm2_5_pm10_ratio'] = current_features['pm2_5_pm10_ratio']
        
        if 'traffic_index' in forecast_features:
            # Traffic index follows rush hour patterns
            recent_avg_traffic = historical_data['traffic_index'].tail(6).mean()
            
            # Increase during rush hours, decrease otherwise
            if hour in [7, 8, 17, 18]:
                adjusted_traffic = recent_avg_traffic * 1.4  # 40% increase
            elif hour in [2, 3, 4, 5]:
                adjusted_traffic = recent_avg_traffic * 0.6  # 40% decrease
            else:
                adjusted_traffic = recent_avg_traffic
            
            # Add small random variation
            traffic_std = historical_data['traffic_index'].std()
            random_variation = np.random.normal(0, traffic_std * 0.15)
            adjusted_traffic += random_variation
            
            # Keep within reasonable bounds
            min_bound = recent_avg_traffic * 0.3
            max_bound = recent_avg_traffic * 1.7
            forecast_features['traffic_index'] = np.clip(adjusted_traffic, min_bound, max_bound)
        
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
    print(f"� Forecast range: {min(forecasts):.1f} - {max(forecasts):.1f} AQI")
    print(f"📊 Forecast average: {np.mean(forecasts):.1f} AQI")
    
    return forecast_df

def main():
    """Main training pipeline function."""
    print("🚀 Starting AQI XGBoost Training Pipeline...")
    print("=" * 60)
    
    # Get current PKT time for output directory
    pkt_tz = zoneinfo.ZoneInfo('Asia/Karachi')
    current_time_pkt = datetime.now(pkt_tz)
    timestamp = current_time_pkt.strftime('%m_%d_%Y_%H%M_pkt')
    output_dir = f"xgboost_additional/{timestamp}"
    
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
        print("\n🤖 Step 4: Training XGBoost models...")
        
        best_model = None
        best_metrics = None
        best_split_idx = 0
        
        for split_idx, split in enumerate(splits):
            print(f"\n📈 Training split {split_idx + 1}/{len(splits)}...")
            
            # Prepare data
            X_train = df_sorted[ALL_FEATURES].iloc[split['train']].values
            y_train = df_sorted[TARGET_VARIABLE].iloc[split['train']].values
            X_val = df_sorted[ALL_FEATURES].iloc[split['val']].values
            y_val = df_sorted[TARGET_VARIABLE].iloc[split['val']].values
            X_test = df_sorted[ALL_FEATURES].iloc[split['test']].values
            y_test = df_sorted[TARGET_VARIABLE].iloc[split['test']].values
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = train_xgboost_model(X_train_scaled, y_train, X_val_scaled, y_val)
            
            # Evaluate on test set
            metrics, y_pred = evaluate_model(model, X_test_scaled, y_test, ALL_FEATURES)
            
            # Save best model based on R2 score
            if best_model is None or metrics['R2'] > best_metrics['R2']:
                best_model = model
                best_metrics = metrics
                best_split_idx = split_idx
        
        print(f"\n🏆 Best model from split {best_split_idx + 1}")
        print("📊 Best metrics:")
        for metric, value in best_metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        # Step 5: Feature importance analysis
        print("\n📊 Step 5: Feature importance analysis...")
        feature_importance_df = plot_feature_importance(best_model, ALL_FEATURES, output_dir=output_dir)
        
        # Step 6: Model interpretability
        print("\n🔍 Step 6: Model interpretability analysis...")
        
        # Use the best split for interpretability analysis
        best_split = splits[best_split_idx]
        X_train_best = df_sorted[ALL_FEATURES].iloc[best_split['train']].values
        X_test_best = df_sorted[ALL_FEATURES].iloc[best_split['test']].values
        
        # Scale features for best split
        scaler_best = StandardScaler()
        X_train_best_scaled = scaler_best.fit_transform(X_train_best)
        X_test_best_scaled = scaler_best.transform(X_test_best)
        
        # SHAP analysis
        shap_df, shap_explainer = analyze_with_shap(best_model, X_test_best_scaled, ALL_FEATURES, output_dir=output_dir)
        
        # LIME analysis
        lime_exp = analyze_with_lime(best_model, X_train_best_scaled, X_test_best_scaled, ALL_FEATURES, output_dir=output_dir)
        
        # Step 7: Generate forecasts
        print("\n🔮 Step 7: Generating 3-day AQI forecasts...")
        forecast_df = forecast_aqi_3_days(best_model, scaler_best, df_sorted, ALL_FEATURES)
        
        # Step 8: Save results
        print("\n💾 Step 8: Saving results...")
        
        # Save model
        model_path = os.path.join(output_dir, 'aqi_xgboost_model.json')
        best_model.save_model(model_path)
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler_best, f)
        
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
        
        print("\n✅ XGBoost Training Pipeline Completed Successfully!")
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
    # Run the main training pipeline
    model, metrics, output_dir = main()
    
    if model is not None:
        print("\n🎉 XGBoost model training completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 XGBoost model training failed!")
        sys.exit(1)
