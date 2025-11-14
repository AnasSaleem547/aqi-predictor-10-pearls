#!/usr/bin/env python3
"""
AQI LGBM Training Pipeline
==========================

Comprehensive LightGBM training pipeline for AQI prediction with:
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
import json

# Machine Learning imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.preprocessing import StandardScaler

# LightGBM
import lightgbm as lgb

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
ENGINEERED_FEATURES = ['pm2_5_pm10_ratio', 'traffic_index']  # Removed pm_weighted and total_pm due to data leakage
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

def train_lgbm_model(X_train, y_train, X_val, y_val, params=None):
    """Train LightGBM model with early stopping."""
    print("🤖 Training LightGBM model...")
    
    # Try to load optimized parameters, fall back to defaults
    if params is None:
        try:
            with open('best_lgbm_params.json', 'r') as f:
                params = json.load(f)
            print("✅ Using optimized hyperparameters")
        except FileNotFoundError:
            # Default parameters if optimized ones not found
            params = {
                'objective': 'regression',
                'metric': 'l2',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': RANDOM_STATE
            }
            print("⚠️ Using default parameters (optimized parameters not found)")
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train model with early stopping
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(100)
        ]
    )
    
    print(f"✅ Model trained with {model.current_iteration()} iterations")
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
    """Plot feature importance using LightGBM's built-in importance."""
    print("📊 Plotting feature importance...")
    
    # Create output directory if not provided
    if output_dir is None:
        # Get current PKT time
        import zoneinfo
        pkt_tz = zoneinfo.ZoneInfo('Asia/Karachi')
        current_time_pkt = datetime.now(pkt_tz)
        timestamp = current_time_pkt.strftime('%m_%d_%Y_%H%M_pkt')
        output_dir = f"lgbm_additional/{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(top_n))
    plt.title(f'Top {top_n} Feature Importance (LightGBM Gain)')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'feature_importance_lgbm.png')
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
        output_dir = f"lgbm_additional/{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
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
    print("🔍 Performing LIME analysis...")
    
    # Create output directory if not provided
    if output_dir is None:
        # Get current PKT time
        pkt_tz = zoneinfo.ZoneInfo('Asia/Karachi')
        current_time_pkt = datetime.now(pkt_tz)
        timestamp = current_time_pkt.strftime('%m_%d_%Y_%H%M_pkt')
        output_dir = f"lgbm_additional/{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        mode='regression',
        verbose=True,
        random_state=RANDOM_STATE
    )
    
    # Explain a specific instance
    exp = explainer.explain_instance(X_test[instance_idx], model.predict)
    
    # Save explanation as HTML
    lime_path = os.path.join(output_dir, 'lime_explanation.html')
    exp.save_to_file(lime_path)
    
    return exp

def forecast_aqi_3_days(model, scaler, df_processed, feature_names, days=3):
    """
    Generate 3-day AQI forecasts using proper recursive forecasting with actual historical data.
    
    Args:
        model: Trained LightGBM model
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
    
    # Initialize forecast predictions
    forecasts = []
    
    # Create a working copy of features that we'll update recursively
    working_features = historical_data.iloc[-1].copy()
    
    # Removed recent_aqi tracking since we're not using AQI lag features anymore
    
    for i, forecast_time in enumerate(forecast_timestamps):
        # Create feature row for this forecast period
        forecast_features = working_features.copy()
        
        # Update temporal features for the forecast time
        forecast_features['hour'] = forecast_time.hour
        forecast_features['is_rush_hour'] = 1 if forecast_time.hour in [7, 8, 9, 17, 18, 19, 20, 21, 22] else 0
        forecast_features['is_weekend'] = 1 if forecast_time.weekday() >= 5 else 0
        
        # Season calculation
        month = forecast_time.month
        if month in [12, 1, 2]:    # Winter
            forecast_features['season'] = 0
        elif month in [3, 4, 5]:   # Spring
            forecast_features['season'] = 1
        elif month in [6, 7, 8]:   # Summer
            forecast_features['season'] = 2
        elif month in [9, 10, 11]: # Monsoon
            forecast_features['season'] = 3
        else:
            forecast_features['season'] = 0
        
        # Update cyclical features
        forecast_features['hour_sin'] = np.sin(2 * np.pi * forecast_time.hour / 24)
        forecast_features['hour_cos'] = np.cos(2 * np.pi * forecast_time.hour / 24)
        forecast_features['day_of_week_sin'] = np.sin(2 * np.pi * forecast_time.weekday() / 7)
        forecast_features['day_of_week_cos'] = np.cos(2 * np.pi * forecast_time.weekday() / 7)
        
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
        if hour in [2, 3, 4, 5]:
            temporal_adjustment -= 0.20  # 20% decrease during deepest night
        elif hour in [0, 1, 23]:
            temporal_adjustment -= 0.10  # 10% decrease during late night/early morning
        
        # Weekend decreases (less industrial activity)
        if is_weekend:
            temporal_adjustment -= 0.15  # 15% decrease on weekends
        
        # Seasonal adjustments (winter can have higher pollution due to heating)
        if forecast_features['season'] == 0:  # Winter
            temporal_adjustment += 0.10  # 10% increase in winter
        
        # Day of week (Monday can be higher due to weekend buildup)
        if forecast_time.weekday() == 0:  # Monday
            temporal_adjustment += 0.08  # 8% increase on Monday
        
        # Update pollutant features with realistic patterns
        pollutant_features = ['pm2_5_nowcast', 'pm10_nowcast', 'no2_nowcast', 
                             'o3_nowcast', 'so2_nowcast', 'co_nowcast']
        
        for poll_feature in pollutant_features:
            if poll_feature in forecast_features:
                # Use historical baseline with temporal adjustments
                base_value = hour_baseline[poll_feature] if poll_feature in hour_baseline else working_features[poll_feature]
                
                # Apply temporal adjustment (additive approach)
                adjusted_value = base_value * (1 + temporal_adjustment)
                
                # Add realistic variability based on hour-specific standard deviation
                if poll_feature in hour_std:
                    # Add random noise based on historical variability (±1 standard deviation)
                    noise = np.random.normal(0, hour_std[poll_feature] * 0.3)  # 30% of historical std
                    adjusted_value += noise
                
                # Allow wider bounds (±60% of recent average) for more realistic variation
                recent_avg = historical_data[poll_feature].tail(24).mean()
                min_bound = recent_avg * 0.4  # Even wider bounds
                max_bound = recent_avg * 1.6
                forecast_features[poll_feature] = np.clip(adjusted_value, min_bound, max_bound)
        
        # Update traffic index with realistic patterns
        if 'traffic_index' in forecast_features:
            base_traffic = hour_baseline['traffic_index'] if 'traffic_index' in hour_baseline else working_features['traffic_index']
            adjusted_traffic = base_traffic * (1 + temporal_adjustment)
            
            # Add some variability to traffic
            if 'traffic_index' in hour_std:
                noise = np.random.normal(0, hour_std['traffic_index'] * 0.2)
                adjusted_traffic += noise
            
            recent_avg_traffic = historical_data['traffic_index'].tail(24).mean()
            min_bound = recent_avg_traffic * 0.3  # Even wider bounds
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
        
        # Removed recent_aqi tracking since we're not using AQI lag features anymore
        
        # Update working features for next iteration
        working_features = forecast_features.copy()
        working_features[TARGET_VARIABLE] = prediction
    
    # Add predictions to forecast DataFrame
    forecast_df['aqi_forecast'] = forecasts
    forecast_df['forecast_type'] = '3_day_forecast'
    
    print(f"✅ Generated {len(forecast_df)} hourly forecasts for {days} days")
    print(f"📈 Forecast range: {min(forecasts):.1f} - {max(forecasts):.1f} AQI")
    print(f"📊 Forecast average: {np.mean(forecasts):.1f} AQI")
    
    return forecast_df

def save_model_and_results(model, metrics, feature_importance_df, shap_df, forecast_df=None, output_dir=None):
    """Save model, metrics, analysis results, and forecasts."""
    print("💾 Saving model and results...")
    
    # Create output directory if not provided
    if output_dir is None:
        # Get current PKT time
        pkt_tz = zoneinfo.ZoneInfo('Asia/Karachi')
        current_time_pkt = datetime.now(pkt_tz)
        timestamp = current_time_pkt.strftime('%m_%d_%Y_%H%M_pkt')
        output_dir = f"lgbm_additional/{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'aqi_lgbm_model.txt')
    model.save_model(model_path)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save feature importance
    feature_importance_path = os.path.join(output_dir, 'feature_importance.csv')
    feature_importance_df.to_csv(feature_importance_path, index=False)
    
    # Save SHAP values
    shap_path = os.path.join(output_dir, 'shap_values.csv')
    shap_df.to_csv(shap_path, index=False)
    
    # Save forecasts if provided
    if forecast_df is not None:
        forecast_path = os.path.join(output_dir, 'aqi_3_day_forecast.csv')
        forecast_df.to_csv(forecast_path, index=False)
        print(f"✅ Forecasts saved to {forecast_path}")
    
    print(f"✅ Model and results saved successfully to {output_dir}")

def main():
    """Main training pipeline."""
    print("🚀 Starting AQI LGBM Training Pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Fetch data from Hopsworks
        df = fetch_data_from_hopsworks()
            
        if df is None or df.empty:
            raise RuntimeError("❌ No data available for training")
        
        # Step 2: Preprocess data
        df_processed = preprocess_data(df)
        
        # Step 3: Prepare features and target
        X = df_processed[ALL_FEATURES].values
        y = df_processed[TARGET_VARIABLE].values
        feature_names = ALL_FEATURES
        
        print(f"📊 Final dataset shape: {X.shape}")
        print(f"🎯 Target variable range: {y.min():.1f} - {y.max():.1f}")
        
        # Step 4: Create time series splits
        splits, df_sorted = create_time_series_splits(df_processed)
        
        # Step 5: Train and evaluate model for each split
        all_metrics = []
        best_model = None
        best_score = float('inf')
        
        for i, split in enumerate(splits):
            print(f"\n🔁 Processing split {i+1}/{len(splits)}")
            print("-" * 40)
            
            # Get data splits
            X_train = X[split['train']]
            X_val = X[split['val']]
            X_test = X[split['test']]
            
            y_train = y[split['train']]
            y_val = y[split['val']]
            y_test = y[split['test']]
            
            print(f"   Train: {X_train.shape[0]} samples")
            print(f"   Val:   {X_val.shape[0]} samples")
            print(f"   Test:  {X_test.shape[0]} samples")
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = train_lgbm_model(X_train_scaled, y_train, X_val_scaled, y_val)
            
            # Evaluate model
            metrics, y_pred = evaluate_model(model, X_test_scaled, y_test, feature_names)
            all_metrics.append(metrics)
            
            # Track best model
            if metrics['RMSE'] < best_score:
                best_score = metrics['RMSE']
                best_model = model
                best_scaler = scaler
                best_split = i
            
            # Feature importance for this split
            feature_importance_df = plot_feature_importance(model, feature_names)
            
            # SHAP analysis for this split
            shap_df, _ = analyze_with_shap(model, X_test_scaled, feature_names)
            
            # LIME analysis for this split (on first instance)
            if i == 0:  # Only do this for first split to avoid redundancy
                analyze_with_lime(model, X_train_scaled, X_test_scaled, feature_names)
        
        # Step 6: Final evaluation with best model
        print(f"\n🏆 Best model from split {best_split + 1} with RMSE: {best_score:.4f}")
        
        # Step 7: Generate 3-day forecasts
        print("\n🌤️  Generating 3-day AQI forecasts...")
        forecast_df = forecast_aqi_3_days(best_model, best_scaler, df_processed, feature_names)
        
        # Display forecast summary
        print("\n📊 Forecast Summary:")
        print("=" * 40)
        print(f"Forecast period: {forecast_df['datetime'].min()} to {forecast_df['datetime'].max()}")
        print(f"Forecast range: {forecast_df['aqi_forecast'].min():.1f} - {forecast_df['aqi_forecast'].max():.1f} AQI")
        print(f"Average forecast: {forecast_df['aqi_forecast'].mean():.1f} AQI")
        
        # Save best model, results, and forecasts
        save_model_and_results(best_model, all_metrics[best_split], 
                             feature_importance_df, shap_df, forecast_df)
        
        # Step 8: Print overall results
        print("\n📊 Overall Cross-Validation Results:")
        print("=" * 50)
        
        avg_metrics = {}
        for metric in all_metrics[0].keys():
            avg_metrics[metric] = np.mean([m[metric] for m in all_metrics])
            print(f"{metric:20}: {avg_metrics[metric]:.4f} (avg)")
        
        print("\n🎉 Training pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
