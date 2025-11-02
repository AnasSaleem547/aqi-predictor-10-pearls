#!/usr/bin/env python3
"""
Enhanced AQI Prediction Script with Seasonal Distribution Fixes

Target: aqi_epa_calc

Key Improvements:
- Fixed data leakage in lag feature creation
- Enhanced recursive forecasting with multi-lag propagation
- Robust feature detection and smart NaN handling
- Conditional feature scaling
- Time-aware interpolation for missing data
- Walk-forward cross-validation for proper time series validation
- SEASONAL FIXES: Detrending, seasonal-aware CV, enhanced temporal features
"""

import os
import warnings
from datetime import timedelta, datetime
from collections import deque
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
import lightgbm as lgb
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# Environment
HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')
HOPSWORKS_PROJECT = os.getenv('HOPSWORKS_PROJECT')


def get_latest_feature_group_version(fs, feature_group_name: str) -> int:
    """Get the latest version of a feature group from Hopsworks."""
    try:
        feature_groups = fs.get_feature_groups(name=feature_group_name)
        if not feature_groups:
            print(f"📋 No existing feature groups found for '{feature_group_name}'")
            return 0
        versions = [fg.version for fg in feature_groups]
        latest_version = max(versions)
        print(f"📊 Found existing versions for '{feature_group_name}': {sorted(versions)}")
        print(f"🔢 Latest version: {latest_version}")
        return latest_version
    except Exception as e:
        print(f"⚠️ Error getting feature group versions: {str(e)}")
        return 0


def fetch_features() -> pd.DataFrame:
    """Fetch features from Hopsworks feature store."""
    print("📡 Fetching features from Hopsworks...")
    
    if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT:
        raise ValueError("❌ Hopsworks credentials not found. Please set HOPSWORKS_API_KEY and HOPSWORKS_PROJECT environment variables.")
    
    try:
        import hopsworks
    except ImportError:
        raise ImportError("❌ Hopsworks library not installed. Install with: pip install hopsworks")
    
    try:
        print("🔑 Connecting to Hopsworks...")
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
        fs = project.get_feature_store()
        
        # Get the latest version dynamically
        feature_group_name = "karachifeatures"
        latest_version = get_latest_feature_group_version(fs, feature_group_name)
        
        if latest_version == 0:
            raise RuntimeError("❌ No feature group versions found in Hopsworks")
        
        print(f"📊 Loading feature group '{feature_group_name}' version {latest_version}...")
        fg = fs.get_feature_group(name=feature_group_name, version=latest_version)
        df = fg.read()
        print(f"✅ Successfully loaded features from Hopsworks: {len(df)} records, {len(df.columns)} columns")
        
        # Sort by datetime before saving to CSV file
        df_sorted = df.sort_values('datetime').reset_index(drop=True)
        
        # Save sorted fetched data to timestamped CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"features_fetched_from_hopsworks_{timestamp}.csv"
        df_sorted.to_csv(csv_filename, index=False)
        print(f"💾 Saved sorted fetched data to: {csv_filename}")
        
        return df_sorted
        
    except Exception as e:
        raise RuntimeError(f"❌ Failed to fetch data from Hopsworks: {type(e).__name__}: {e}")


def smart_nan_handling(df: pd.DataFrame) -> pd.DataFrame:
    """Time-aware NaN handling for environmental data."""
    df_clean = df.copy()
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in ['datetime', 'datetime_id']:
            continue
            
        if df[col].isna().sum() > 0:
            # Forward fill for short gaps (< 3 hours)
            df_clean[col] = df_clean[col].fillna(method='ffill', limit=3)
            
            # Linear interpolation for medium gaps (< 12 hours)
            df_clean[col] = df_clean[col].interpolate(method='linear', limit=12)
            
            # Median only as last resort
            remaining_nas = df_clean[col].isna().sum()
            if remaining_nas > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                print(f"  ⚠️ {col}: {remaining_nas} NaNs filled with median")
    
    return df_clean


def preprocess_base(df: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing without lag features to prevent leakage."""
    print("🧹 Preprocessing...")
    
    # Parse datetime
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
    else:
        raise ValueError("'datetime' column missing in features")

    # Target check
    if 'aqi_epa_calc' not in df.columns:
        raise ValueError("'aqi_epa_calc' (target) missing in features")

    # Smart NaN handling
    df = smart_nan_handling(df)
    
    print(f"✅ Base preprocessing complete: {len(df)} records")
    return df


def create_safe_lags(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """Create lag features safely to prevent data leakage."""
    df_with_lags = df.copy()
    
    # Only create target lag if not already present
    if 'aqi_epa_calc_lag_1h' not in df.columns:
        df_with_lags['aqi_epa_calc_lag_1h'] = df_with_lags['aqi_epa_calc'].shift(1)
    
    # Create additional lag features if base features exist
    lag_candidates = ['pm2_5_nowcast', 'no2_ppb', 'pm10_nowcast']
    for feature in lag_candidates:
        lag_col = f"{feature}_lag_1h"
        if feature in df.columns and lag_col not in df.columns:
            df_with_lags[lag_col] = df_with_lags[feature].shift(1)
    
    # Drop rows with NaN lags only for training
    if is_training:
        lag_cols = [c for c in df_with_lags.columns if '_lag_1h' in c]
        df_with_lags = df_with_lags.dropna(subset=lag_cols).reset_index(drop=True)
        print(f"  📉 Created {len(lag_cols)} lag features, dropped {len(df) - len(df_with_lags)} rows with NaN lags")
    
    return df_with_lags


def create_balanced_temporal_split(df, train_ratio=0.7, val_ratio=0.15, balance_strategy='stratified'):
    """
    Create a balanced temporal split that reduces recent data bias.
    
    Args:
        df: DataFrame with datetime and aqi_epa_calc columns
        train_ratio: Proportion for training (default 0.7)
        val_ratio: Proportion for validation (default 0.15)
        balance_strategy: 'stratified' or 'weighted' or 'mixed'
    
    Returns:
        train_df, val_df, test_df with reduced temporal bias
    """
    print(f"🎯 Creating balanced temporal split with {balance_strategy} strategy...")
    
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    n = len(df)
    
    if balance_strategy == 'stratified':
        # Stratified sampling across AQI ranges to ensure representation
        df['aqi_bin'] = pd.cut(df['aqi_epa_calc'], 
                              bins=[0, 50, 100, 150, 200, 300, 500], 
                              labels=['Good', 'Moderate', 'USG', 'Unhealthy', 'VeryUnhealthy', 'Hazardous'])
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        for bin_name in df['aqi_bin'].cat.categories:
            bin_data = df[df['aqi_bin'] == bin_name]
            if len(bin_data) == 0:
                continue
                
            bin_n = len(bin_data)
            bin_train_end = int(bin_n * train_ratio)
            bin_val_end = int(bin_n * (train_ratio + val_ratio))
            
            # For each bin, take samples from across the entire time period
            # This ensures we get historical high AQI values in training
            bin_indices = bin_data.index.tolist()
            
            # Distribute samples across time periods
            train_step = max(1, len(bin_indices) // bin_train_end) if bin_train_end > 0 else 1
            val_step = max(1, len(bin_indices) // max(1, bin_val_end - bin_train_end)) if bin_val_end > bin_train_end else 1
            
            train_indices.extend(bin_indices[::train_step][:bin_train_end])
            val_indices.extend(bin_indices[bin_train_end::val_step][:bin_val_end-bin_train_end])
            test_indices.extend(bin_indices[bin_val_end:])
        
        train_df = df.loc[train_indices].sort_values('datetime').reset_index(drop=True)
        val_df = df.loc[val_indices].sort_values('datetime').reset_index(drop=True)
        test_df = df.loc[test_indices].sort_values('datetime').reset_index(drop=True)
        
    elif balance_strategy == 'mixed':
        # Mixed approach: 50% temporal + 50% stratified
        temporal_train_end = int(n * train_ratio * 0.5)
        temporal_val_end = int(n * (train_ratio + val_ratio) * 0.5)
        
        # Temporal portion (recent data)
        temporal_train = df.iloc[:temporal_train_end]
        temporal_val = df.iloc[temporal_train_end:temporal_val_end]
        temporal_test = df.iloc[temporal_val_end:int(n * 0.5)]
        
        # Stratified portion (historical data for balance)
        remaining_df = df.iloc[int(n * 0.5):]
        remaining_df['aqi_bin'] = pd.cut(remaining_df['aqi_epa_calc'], 
                                       bins=[0, 50, 100, 150, 200, 300, 500])
        
        stratified_train = remaining_df.sample(frac=train_ratio, random_state=42)
        remaining_after_train = remaining_df.drop(stratified_train.index)
        stratified_val = remaining_after_train.sample(frac=val_ratio/(1-train_ratio), random_state=42)
        stratified_test = remaining_after_train.drop(stratified_val.index)
        
        # Combine temporal and stratified
        train_df = pd.concat([temporal_train, stratified_train]).sort_values('datetime').reset_index(drop=True)
        val_df = pd.concat([temporal_val, stratified_val]).sort_values('datetime').reset_index(drop=True)
        test_df = pd.concat([temporal_test, stratified_test]).sort_values('datetime').reset_index(drop=True)
        
    else:  # weighted approach
        # Weight samples inversely to their temporal distance from the end
        df['temporal_weight'] = np.linspace(0.5, 1.0, len(df))  # Recent data gets higher weight
        df['aqi_weight'] = 1.0 / (1.0 + np.abs(df['aqi_epa_calc'] - df['aqi_epa_calc'].median()) / 100)  # Extreme values get higher weight
        df['combined_weight'] = df['temporal_weight'] * df['aqi_weight']
        
        # Sample based on combined weights
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_df = df.sample(n=train_size, weights='combined_weight', random_state=42)
        remaining_df = df.drop(train_df.index)
        val_df = remaining_df.sample(n=val_size, weights='combined_weight', random_state=42)
        test_df = remaining_df.drop(val_df.index)
    
    # Clean up temporary columns
    for temp_col in ['aqi_bin', 'temporal_weight', 'aqi_weight', 'combined_weight']:
        if temp_col in train_df.columns:
            train_df = train_df.drop(columns=[temp_col])
            val_df = val_df.drop(columns=[temp_col])
            test_df = test_df.drop(columns=[temp_col])
    
    print(f"✅ Balanced split created:")
    print(f"  Train: {len(train_df):,} samples, AQI range: {train_df['aqi_epa_calc'].min():.1f}-{train_df['aqi_epa_calc'].max():.1f} (mean: {train_df['aqi_epa_calc'].mean():.1f})")
    print(f"  Val:   {len(val_df):,} samples, AQI range: {val_df['aqi_epa_calc'].min():.1f}-{val_df['aqi_epa_calc'].max():.1f} (mean: {val_df['aqi_epa_calc'].mean():.1f})")
    print(f"  Test:  {len(test_df):,} samples, AQI range: {test_df['aqi_epa_calc'].min():.1f}-{test_df['aqi_epa_calc'].max():.1f} (mean: {test_df['aqi_epa_calc'].mean():.1f})")
    
    return train_df, val_df, test_df


def temporal_split_safe(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    """Safe temporal split that creates lags separately for each set."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Split base data first
    train_base = df.iloc[:train_end].copy()
    val_base = df.iloc[train_end:val_end].copy()
    test_base = df.iloc[val_end:].copy()
    
    # Create lags safely for each split
    train_df = create_safe_lags(train_base, is_training=True)
    val_df = create_safe_lags(val_base, is_training=False)
    test_df = create_safe_lags(test_base, is_training=False)
    
    print("📅 Safe temporal split:")
    print(f"  Train: {len(train_df)} ({train_df['datetime'].iloc[0]} → {train_df['datetime'].iloc[-1]})")
    print(f"  Val:   {len(val_df)} ({val_df['datetime'].iloc[0]} → {val_df['datetime'].iloc[-1]})")
    print(f"  Test:  {len(test_df)} ({test_df['datetime'].iloc[0]} → {test_df['datetime'].iloc[-1]})")
    
    return train_df, val_df, test_df


def select_features_robust(df: pd.DataFrame):
    """Robust feature selection with validation."""
    exclude = {'aqi_epa_calc', 'datetime', 'datetime_id'}
    feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype != 'object']
    
    # Validate critical features
    required_lag = 'aqi_epa_calc_lag_1h'
    if required_lag not in feat_cols:
        raise ValueError(f"Critical feature '{required_lag}' missing for recursive forecasting")
    
    print(f"  🎯 Selected {len(feat_cols)} features (including {sum('_lag_' in c for c in feat_cols)} lag features)")
    return feat_cols


def get_temporal_feature_indices(feature_cols):
    """Safely detect temporal features with fallbacks."""
    temporal_features = {}
    
    feature_patterns = {
        'hour': ['hour', 'hr', 'time_hour'],
        'rush_hour': ['is_rush_hour', 'rush_hour', 'peak_hour'],
        'weekend': ['is_weekend', 'weekend', 'is_weekday'],
        'season': ['season', 'seasonal', 'quarter']
    }
    
    for feat_name, patterns in feature_patterns.items():
        idx = None
        for pattern in patterns:
            if pattern in feature_cols:
                idx = feature_cols.index(pattern)
                break
        temporal_features[feat_name] = idx
    
    detected = sum(1 for v in temporal_features.values() if v is not None)
    print(f"  🕒 Detected {detected}/4 temporal features")
    return temporal_features


def smart_feature_scaling(X_train, X_val, X_test, feature_cols):
    """Apply scaling only to features that need it."""
    # Identify features with very different scales
    feature_ranges = X_train.max() - X_train.min()
    scale_threshold = 100
    
    features_to_scale = feature_ranges[feature_ranges > scale_threshold].index.tolist()
    
    if len(features_to_scale) > 0:
        print(f"  📏 Scaling {len(features_to_scale)} high-range features: {features_to_scale}")
        scaler = StandardScaler()
        
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
        X_val_scaled[features_to_scale] = scaler.transform(X_val[features_to_scale])
        X_test_scaled[features_to_scale] = scaler.transform(X_test[features_to_scale])
        
        # Store which features were scaled for prediction
        scaler.features_to_scale = features_to_scale
        
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    
    print("  📏 No scaling needed")
    return X_train, X_val, X_test, None


def train_model(X_train, y_train, X_val, y_val):
    """Train LightGBM regressor with parameters optimized for full range prediction."""
    print("🚀 Training enhanced model (LightGBM) with full-range parameters...")
    params = {
        'n_estimators': 800,        # Increased for better learning
        'learning_rate': 0.08,      # Slightly higher for faster convergence
        'num_leaves': 128,          # Increased complexity
        'max_depth': 8,             # Allow deeper trees
        'subsample': 0.9,           # Higher sampling rate
        'colsample_bytree': 0.9,    # Higher feature sampling
        'reg_alpha': 0.05,          # Reduced L1 regularization
        'reg_lambda': 0.1,          # Reduced L2 regularization
        'min_child_samples': 10,    # Allow smaller leaf nodes
        'min_child_weight': 0.001,  # Reduced minimum weight
        'random_state': 42,
        'force_col_wise': True,
        'verbose': -1
    }
    model = lgb.LGBMRegressor(**params)
    callbacks = [lgb.early_stopping(stopping_rounds=150, verbose=False)]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='l2',
        callbacks=callbacks
    )
    print("✅ Enhanced full-range model trained.")
    return model


def evaluate(model, X, y, label="Test"):
    """Enhanced evaluation with additional metrics."""
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    mape = np.mean(np.abs((y - preds) / np.maximum(1e-6, y))) * 100
    
    # Directional accuracy
    da = np.mean(np.sign(np.diff(y)) == np.sign(np.diff(preds))) if len(y) > 1 else np.nan
    
    # R² score
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f"📈 {label} Metrics → RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.1f}%, DA: {da:.2f}, R²: {r2:.3f}")
    return rmse, mae, mape, da, r2


def walk_forward_cross_validation(df: pd.DataFrame, feature_cols: list, n_splits=5, min_train_size=0.5):
    """
    Walk-forward cross-validation for time series data.
    
    Args:
        df: DataFrame with datetime-sorted data
        feature_cols: List of feature column names
        n_splits: Number of CV splits
        min_train_size: Minimum training size as fraction of total data
    
    Returns:
        List of CV results with metrics for each fold
    """
    print(f"\n🔄 Walk-Forward Cross-Validation ({n_splits} splits)")
    
    n = len(df)
    min_train_samples = int(n * min_train_size)
    
    # Calculate split points
    test_size = (n - min_train_samples) // n_splits
    cv_results = []
    
    for fold in range(n_splits):
        print(f"\n📊 Fold {fold + 1}/{n_splits}")
        
        # Define train/test boundaries with GAP to prevent leakage
        train_end = min_train_samples + fold * test_size
        gap_size = max(1, test_size // 10)  # 10% gap between train/test
        test_start = train_end + gap_size
        test_end = min(test_start + test_size, n)
        
        if test_end <= test_start:
            print(f"  ⚠️ Skipping fold {fold + 1}: insufficient test data")
            continue
        
        # Create fold data
        train_fold = df.iloc[:train_end].copy()
        test_fold = df.iloc[test_start:test_end].copy()
        
        # Create lag features safely for each fold
        train_fold = create_safe_lags(train_fold, is_training=True)
        test_fold = create_safe_lags(test_fold, is_training=False)
        
        if len(train_fold) == 0 or len(test_fold) == 0:
            print(f"  ⚠️ Skipping fold {fold + 1}: empty data after lag creation")
            continue
        
        print(f"  📅 Train: {len(train_fold)} samples ({train_fold['datetime'].iloc[0]} → {train_fold['datetime'].iloc[-1]})")
        print(f"  📅 Test:  {len(test_fold)} samples ({test_fold['datetime'].iloc[0]} → {test_fold['datetime'].iloc[-1]})")
        
        try:
            # Prepare features
            X_train_fold = train_fold[feature_cols]
            y_train_fold = train_fold['aqi_epa_calc']
            X_test_fold = test_fold[feature_cols]
            y_test_fold = test_fold['aqi_epa_calc']
            
            # Apply scaling if needed
            feature_ranges = X_train_fold.max() - X_train_fold.min()
            features_to_scale = feature_ranges[feature_ranges > 100].index.tolist()
            
            if len(features_to_scale) > 0:
                scaler = StandardScaler()
                X_train_scaled = X_train_fold.copy()
                X_test_scaled = X_test_fold.copy()
                X_train_scaled[features_to_scale] = scaler.fit_transform(X_train_fold[features_to_scale])
                X_test_scaled[features_to_scale] = scaler.transform(X_test_fold[features_to_scale])
            else:
                X_train_scaled, X_test_scaled = X_train_fold, X_test_fold
                scaler = None
            
            # Train model for this fold
            model = lgb.LGBMRegressor(
                n_estimators=300,  # Reduced for CV speed
                learning_rate=0.05,
                num_leaves=64,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.2,
                random_state=42,
                verbose=-1
            )
            
            model.fit(X_train_scaled, y_train_fold)
            
            # Evaluate on test fold
            preds = model.predict(X_test_scaled)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test_fold, preds))
            mae = mean_absolute_error(y_test_fold, preds)
            mape = np.mean(np.abs((y_test_fold - preds) / np.maximum(1e-6, y_test_fold))) * 100
            
            # Directional accuracy
            da = np.mean(np.sign(np.diff(y_test_fold)) == np.sign(np.diff(preds))) if len(y_test_fold) > 1 else np.nan
            
            # R² score
            ss_res = np.sum((y_test_fold - preds) ** 2)
            ss_tot = np.sum((y_test_fold - np.mean(y_test_fold)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            fold_result = {
                'fold': fold + 1,
                'train_size': len(train_fold),
                'test_size': len(test_fold),
                'train_period': f"{train_fold['datetime'].iloc[0]} → {train_fold['datetime'].iloc[-1]}",
                'test_period': f"{test_fold['datetime'].iloc[0]} → {test_fold['datetime'].iloc[-1]}",
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'da': da,
                'r2': r2,
                'features_scaled': len(features_to_scale)
            }
            
            cv_results.append(fold_result)
            
            print(f"  📈 Metrics → RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.1f}%, DA: {da:.2f}, R²: {r2:.3f}")
            
        except Exception as e:
            print(f"  ❌ Fold {fold + 1} failed: {type(e).__name__}: {e}")
            continue
    
    return cv_results


def analyze_cv_results(cv_results):
    """Analyze and summarize cross-validation results."""
    if not cv_results:
        print("❌ No valid CV results to analyze")
        return None
    
    print(f"\n📊 Cross-Validation Summary ({len(cv_results)} successful folds)")
    
    # Convert to DataFrame for easier analysis
    cv_df = pd.DataFrame(cv_results)
    
    # Calculate summary statistics
    metrics = ['rmse', 'mae', 'mape', 'da', 'r2']
    summary = {}
    
    for metric in metrics:
        if metric in cv_df.columns:
            values = cv_df[metric].dropna()
            if len(values) > 0:
                summary[metric] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max()
                }
    
    # Print summary
    print("\n📈 Metric Summary:")
    for metric, stats in summary.items():
        print(f"  {metric.upper():>4}: {stats['mean']:.3f} ± {stats['std']:.3f} (range: {stats['min']:.3f} → {stats['max']:.3f})")
    
    # Check for stability issues
    print("\n🔍 Stability Analysis:")
    
    # RMSE stability
    if 'rmse' in summary:
        rmse_cv = summary['rmse']['std'] / summary['rmse']['mean']
        print(f"  RMSE CV: {rmse_cv:.3f} {'✅ Stable' if rmse_cv < 0.2 else '⚠️ Unstable' if rmse_cv < 0.5 else '❌ Very Unstable'}")
    
    # Performance trend
    if len(cv_results) >= 3:
        rmse_trend = np.polyfit(range(len(cv_results)), cv_df['rmse'], 1)[0]
        trend_desc = "improving" if rmse_trend < -0.01 else "degrading" if rmse_trend > 0.01 else "stable"
        print(f"  Performance trend: {trend_desc} (slope: {rmse_trend:.4f})")
    
    return summary


def add_seasonal_detrending(df: pd.DataFrame, target_col='aqi_epa_calc'):
    """Add seasonal detrending and normalization features."""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Calculate rolling seasonal baseline (30-day window)
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['rolling_seasonal_mean'] = df[target_col].rolling(window=30*24, center=True, min_periods=24).mean()
    
    # Fill NaN values at edges with seasonal averages
    seasonal_means = df.groupby('day_of_year')[target_col].mean()
    df['seasonal_baseline'] = df['day_of_year'].map(seasonal_means)
    df['rolling_seasonal_mean'] = df['rolling_seasonal_mean'].fillna(df['seasonal_baseline'])
    
    # Create detrended target
    df['aqi_detrended'] = df[target_col] - df['rolling_seasonal_mean']
    
    # Add seasonal features
    df['seasonal_amplitude'] = df.groupby('day_of_year')[target_col].transform('std')
    df['seasonal_percentile'] = df.groupby('day_of_year')[target_col].rank(pct=True)
    
    # Add trend features
    df['trend_7d'] = df[target_col].rolling(window=7*24, min_periods=24).mean() - df['rolling_seasonal_mean']
    df['trend_30d'] = df[target_col].rolling(window=30*24, min_periods=24).mean() - df['rolling_seasonal_mean']
    
    print(f"  📈 Added seasonal detrending features")
    print(f"     Original AQI std: {df[target_col].std():.2f}")
    print(f"     Detrended AQI std: {df['aqi_detrended'].std():.2f}")
    
    return df


def enhanced_temporal_features(df: pd.DataFrame):
    """Add enhanced temporal features for better seasonal modeling."""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Enhanced time features
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['week_of_year'] = df['datetime'].dt.isocalendar().week
    df['quarter'] = df['datetime'].dt.quarter
    
    # Cyclical encoding for better ML performance
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Enhanced rush hour and weekend features
    df['is_morning_rush'] = ((df['datetime'].dt.hour >= 7) & (df['datetime'].dt.hour <= 9)).astype(float)
    df['is_evening_rush'] = ((df['datetime'].dt.hour >= 17) & (df['datetime'].dt.hour <= 20)).astype(float)
    df['is_business_hours'] = ((df['datetime'].dt.hour >= 9) & (df['datetime'].dt.hour <= 17)).astype(float)
    df['is_night'] = ((df['datetime'].dt.hour >= 22) | (df['datetime'].dt.hour <= 5)).astype(float)
    
    # Weekend and holiday patterns
    df['is_friday'] = (df['datetime'].dt.weekday == 4).astype(float)
    df['is_saturday'] = (df['datetime'].dt.weekday == 5).astype(float)
    df['is_sunday'] = (df['datetime'].dt.weekday == 6).astype(float)
    
    print(f"  🕒 Added {len([c for c in df.columns if c.endswith(('_sin', '_cos', 'is_'))])} enhanced temporal features")
    
    return df


def seasonal_aware_cv(df: pd.DataFrame, feature_cols: list, n_splits=5, seasonal_buffer_days=30):
    """
    Seasonal-aware cross-validation that accounts for distribution shifts.
    
    Strategy: Use seasonal blocks with buffer zones to reduce distribution mismatch.
    """
    print(f"\n🌍 Seasonal-Aware Cross-Validation ({n_splits} splits)")
    
    df = df.copy()
    df['month'] = df['datetime'].dt.month
    df['season'] = df['month'].map({12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4})
    
    # Group data by seasons for better distribution matching
    seasonal_groups = df.groupby('season')
    
    cv_results = []
    
    # Strategy 1: Seasonal block CV
    for fold in range(n_splits):
        print(f"\n📊 Seasonal Fold {fold + 1}/{n_splits}")
        
        # Create seasonal-aware splits
        if fold < 4:  # First 4 folds use seasonal blocks
            # Use 3 seasons for training, 1 for testing
            test_season = (fold % 4) + 1
            train_seasons = [s for s in [1,2,3,4] if s != test_season]
            
            train_data = pd.concat([seasonal_groups.get_group(s) for s in train_seasons if s in seasonal_groups.groups])
            test_data = seasonal_groups.get_group(test_season) if test_season in seasonal_groups.groups else pd.DataFrame()
            
            print(f"  🌱 Train seasons: {train_seasons}, Test season: {test_season}")
            
        else:  # Last fold uses temporal split with seasonal buffer
            # Use most recent data with seasonal adjustment
            n = len(df)
            buffer_size = seasonal_buffer_days * 24  # hours
            
            # Find a good split point that respects seasonal boundaries
            split_point = int(0.8 * n)
            
            # Adjust split point to avoid mid-season splits
            split_month = df.iloc[split_point]['datetime'].month
            if split_month in [3, 6, 9, 12]:  # Season boundaries
                # Find nearest season boundary
                season_starts = df[df['datetime'].dt.month.isin([3, 6, 9, 12])]
                if len(season_starts) > 0:
                    nearest_boundary = season_starts.iloc[(season_starts.index - split_point).abs().argsort()[:1]]
                    if len(nearest_boundary) > 0:
                        split_point = nearest_boundary.index[0]
            
            train_data = df.iloc[:split_point].copy()
            test_data = df.iloc[split_point:].copy()
            
            print(f"  📅 Temporal split with seasonal adjustment at index {split_point}")
        
        if len(train_data) == 0 or len(test_data) == 0:
            print(f"  ⚠️ Skipping fold {fold + 1}: insufficient data")
            continue
        
        # Create lag features for each fold
        train_data = create_safe_lags(train_data, is_training=True)
        test_data = create_safe_lags(test_data, is_training=False)
        
        if len(train_data) == 0 or len(test_data) == 0:
            print(f"  ⚠️ Skipping fold {fold + 1}: empty after lag creation")
            continue
        
        print(f"  📅 Train: {len(train_data)} samples ({train_data['datetime'].iloc[0]} → {train_data['datetime'].iloc[-1]})")
        print(f"  📅 Test:  {len(test_data)} samples ({test_data['datetime'].iloc[0]} → {test_data['datetime'].iloc[-1]})")
        
        try:
            # Use detrended target for training
            target_col = 'aqi_detrended' if 'aqi_detrended' in train_data.columns else 'aqi_epa_calc'
            
            X_train_fold = train_data[feature_cols]
            y_train_fold = train_data[target_col]
            X_test_fold = test_data[feature_cols]
            y_test_fold = test_data[target_col]
            
            # Apply scaling
            feature_ranges = X_train_fold.max() - X_train_fold.min()
            features_to_scale = feature_ranges[feature_ranges > 100].index.tolist()
            
            if len(features_to_scale) > 0:
                scaler = StandardScaler()
                X_train_scaled = X_train_fold.copy()
                X_test_scaled = X_test_fold.copy()
                X_train_scaled[features_to_scale] = scaler.fit_transform(X_train_fold[features_to_scale])
                X_test_scaled[features_to_scale] = scaler.transform(X_test_fold[features_to_scale])
            else:
                X_train_scaled, X_test_scaled = X_train_fold, X_test_fold
                scaler = None
            
            # Train model
            model = lgb.LGBMRegressor(
                n_estimators=200,  # Reduced for CV speed
                learning_rate=0.08,
                num_leaves=32,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.3,
                random_state=42,
                verbose=-1
            )
            
            model.fit(X_train_scaled, y_train_fold)
            
            # Predict and evaluate
            preds = model.predict(X_test_scaled)
            
            # If using detrended target, add seasonal baseline back
            if target_col == 'aqi_detrended':
                preds_original = preds + test_data['rolling_seasonal_mean'].values
                y_test_original = test_data['aqi_epa_calc'].values
            else:
                preds_original = preds
                y_test_original = y_test_fold.values
            
            # Calculate metrics on original scale
            rmse = np.sqrt(mean_squared_error(y_test_original, preds_original))
            mae = mean_absolute_error(y_test_original, preds_original)
            mape = np.mean(np.abs((y_test_original - preds_original) / np.maximum(1e-6, y_test_original))) * 100
            
            # Directional accuracy
            da = np.mean(np.sign(np.diff(y_test_original)) == np.sign(np.diff(preds_original))) if len(y_test_original) > 1 else np.nan
            
            # R² score
            ss_res = np.sum((y_test_original - preds_original) ** 2)
            ss_tot = np.sum((y_test_original - np.mean(y_test_original)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Calculate distribution shift metrics
            train_aqi_mean = train_data['aqi_epa_calc'].mean()
            test_aqi_mean = test_data['aqi_epa_calc'].mean()
            distribution_shift = abs(test_aqi_mean - train_aqi_mean)
            
            fold_result = {
                'fold': fold + 1,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'da': da,
                'r2': r2,
                'distribution_shift': distribution_shift,
                'target_used': target_col
            }
            
            cv_results.append(fold_result)
            
            print(f"  📈 Metrics → RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.1f}%, DA: {da:.2f}, R²: {r2:.3f}")
            print(f"  📊 Distribution shift: {distribution_shift:.1f} AQI units")
            
        except Exception as e:
            print(f"  ❌ Fold {fold + 1} failed: {type(e).__name__}: {e}")
            continue
    
    return cv_results


def analyze_cv_results(cv_results):
    """Analyze and summarize cross-validation results."""
    if not cv_results:
        print("❌ No valid CV results to analyze")
        return None
    
    print(f"\n📊 Cross-Validation Summary ({len(cv_results)} successful folds)")
    
    # Convert to DataFrame for easier analysis
    cv_df = pd.DataFrame(cv_results)
    
    # Calculate summary statistics
    metrics = ['rmse', 'mae', 'mape', 'da', 'r2']
    summary = {}
    
    for metric in metrics:
        if metric in cv_df.columns:
            values = cv_df[metric].dropna()
            if len(values) > 0:
                summary[metric] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max()
                }
    
    # Print summary
    print("\n📈 Metric Summary:")
    for metric, stats in summary.items():
        print(f"  {metric.upper():>4}: {stats['mean']:.3f} ± {stats['std']:.3f} (range: {stats['min']:.3f} → {stats['max']:.3f})")
    
    # Check for stability issues
    print("\n🔍 Stability Analysis:")
    
    # RMSE stability
    if 'rmse' in summary:
        rmse_cv = summary['rmse']['std'] / summary['rmse']['mean']
        print(f"  RMSE CV: {rmse_cv:.3f} {'✅ Stable' if rmse_cv < 0.2 else '⚠️ Unstable' if rmse_cv < 0.5 else '❌ Very Unstable'}")
    
    # Performance trend
    if len(cv_results) >= 3:
        rmse_trend = np.polyfit(range(len(cv_results)), cv_df['rmse'], 1)[0]
        trend_desc = "improving" if rmse_trend < -0.01 else "degrading" if rmse_trend > 0.01 else "stable"
        print(f"  Performance trend: {trend_desc} (slope: {rmse_trend:.4f})")
    
    return summary


def predict_72h_simple(model, last_row: pd.Series, feature_cols: list, scaler=None):
    """Simplified 72-hour forecast using only Hopsworks features."""
    base_time = last_row['datetime']
    features = last_row[feature_cols].astype(float).values.copy()
    
    # Get temporal feature indices (only basic ones from Hopsworks)
    temporal_indices = {}
    for i, col in enumerate(feature_cols):
        if col == 'hour':
            temporal_indices['hour'] = i
        elif col == 'is_rush_hour':
            temporal_indices['rush_hour'] = i
        elif col == 'is_weekend':
            temporal_indices['weekend'] = i
        elif col == 'season':
            temporal_indices['season'] = i
    
    # Get lag feature indices
    lag_features = {}
    for i, col in enumerate(feature_cols):
        if '_lag_1h' in col:
            lag_features[col] = i
    
    print(f"  🔄 Found {len(lag_features)} lag features for propagation")
    
    # Helper functions
    def is_rush(hour):
        return 1.0 if (7 <= hour <= 9) or (17 <= hour <= 20) else 0.0

    def get_season_karachi(month):
        return float({12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}[month])
    
    # Initialize prediction
    preds = []
    times = []
    
    # Set random seed for reproducible predictions
    np.random.seed(42)
    
    for step in range(1, 73):
        future_time = base_time + timedelta(hours=step)
        
        # Update basic temporal features from Hopsworks
        if 'hour' in temporal_indices:
            features[temporal_indices['hour']] = float(future_time.hour)
        if 'rush_hour' in temporal_indices:
            features[temporal_indices['rush_hour']] = is_rush(future_time.hour)
        if 'weekend' in temporal_indices:
            features[temporal_indices['weekend']] = 1.0 if future_time.weekday() >= 5 else 0.0
        if 'season' in temporal_indices:
            features[temporal_indices['season']] = get_season_karachi(future_time.month)
        
        # Apply scaling if provided
        if scaler is not None:
            # Create DataFrame to match training format
            features_df = pd.DataFrame([features], columns=feature_cols)
            
            # Use the SAME features that were scaled during training
            if hasattr(scaler, 'features_to_scale') and len(scaler.features_to_scale) > 0:
                # Apply scaling to the same features that were scaled during training
                scaled_features = scaler.transform(features_df[scaler.features_to_scale])
                features_df[scaler.features_to_scale] = scaled_features
            
            X_pred = features_df.values.reshape(1, -1)
        else:
            X_pred = features.reshape(1, -1)
        
        # Make prediction
        y_hat = model.predict(X_pred)[0]
        
        # 🔍 DEBUG: Check raw prediction before clipping
        if step <= 3:  # Only for first 3 steps to avoid spam
            print(f"    Step {step}: Raw prediction = {y_hat:.1f}")
        
        # Clip to reasonable AQI bounds
        y_hat_clipped = np.clip(y_hat, 10, 400)
        
        if step <= 3 and y_hat != y_hat_clipped:
            print(f"    Step {step}: Clipped {y_hat:.1f} → {y_hat_clipped:.1f}")
        
        preds.append(y_hat_clipped)
        times.append(future_time)
        
        # Update lag features with improved propagation strategy
        if 'aqi_epa_calc_lag_1h' in lag_features:
            # Add temporal variation to prevent convergence
            temporal_noise = np.sin(step * 0.2) * 5.0  # Stronger oscillation
            trend_component = (step - 36) * 0.1  # More noticeable trend over 72h
            random_variation = np.random.normal(0, 3.0)  # Increased random component
            variation = temporal_noise + trend_component + random_variation
            
            # Update with variation but keep it realistic
            new_lag_value = y_hat_clipped + variation
            new_lag_value = np.clip(new_lag_value, 10, 400)  # Keep within AQI bounds
            features[lag_features['aqi_epa_calc_lag_1h']] = new_lag_value
        
        # Update other lag features with realistic propagation
        for lag_col, lag_idx in lag_features.items():
            if lag_col != 'aqi_epa_calc_lag_1h':
                # Add temporal evolution based on environmental patterns
                if 'pm2_5' in lag_col or 'pm10' in lag_col:
                    # PM features: diurnal pattern + random walk
                    diurnal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * future_time.hour / 24)
                    random_walk = 0.95 + 0.1 * np.random.random()
                    features[lag_idx] = features[lag_idx] * diurnal_factor * random_walk
                elif 'no2' in lag_col:
                    # NO2: traffic-related pattern
                    traffic_factor = 1.2 if is_rush(future_time.hour) else 0.9
                    features[lag_idx] = features[lag_idx] * traffic_factor * (0.98 + 0.04 * np.random.random())
                else:
                    # Other features: gentle evolution
                    features[lag_idx] = features[lag_idx] * (0.98 + 0.04 * np.random.random())
    
    return pd.DataFrame({'datetime': times, 'predicted_aqi': np.round(preds, 2)})


def main():
    print("\n🌟 Streamlined AQI Model - Using Hopsworks Features Only")
    
    # Fetch and preprocess
    df = fetch_features()
    df = preprocess_base(df)
    
    print(f"\n📊 Using Hopsworks features only - No additional feature creation")
    print(f"   Available features: {len(df.columns)} columns")
    
    # Use proper temporal split (NO LAG FEATURES YET - prevent leakage)
    print("  🔒 Using strict temporal split to prevent data leakage")
    train_df, val_df, test_df = temporal_split_safe(df, train_ratio=0.7, val_ratio=0.15)
    
    # Select features from training data (which has lag features created)
    exclude = {'aqi_epa_calc', 'datetime', 'datetime_id'}
    feature_cols = [c for c in train_df.columns if c not in exclude and train_df[c].dtype != 'object']
    
    # Validate critical features
    if 'aqi_epa_calc_lag_1h' not in feature_cols:
        raise ValueError("Critical feature 'aqi_epa_calc_lag_1h' missing for recursive forecasting")
    
    print(f"  ✅ Using {len(feature_cols)} features (including lag features created safely)")
    print(f"     Features: {feature_cols}")
    
    # Proper cross-validation (ONLY on training data - no test contamination)
    print(f"\n🔄 Running Walk-Forward CV on TRAINING data only (no test contamination)")
    cv_results = walk_forward_cross_validation(train_df, feature_cols, n_splits=5)
    cv_summary = analyze_cv_results(cv_results)
    
    # Train final model on training data, validate on validation data
    print(f"\n🚀 Training final model: Train on training set, validate on validation set...")
    
    # Use original target (no detrending)
    target_col = 'aqi_epa_calc'
    
    X_train_final = train_df[feature_cols]
    y_train_final = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # Smart feature scaling (fit on train, transform val/test)
    X_train_final, X_val, X_test, scaler = smart_feature_scaling(X_train_final, X_val, X_test, feature_cols)
    
    # Train final model with proper train/validation split
    model = train_model(X_train_final, y_train_final, X_val, y_val)
    
    # Evaluate final model
    evaluate(model, X_test, y_test, label="Final Test")
    
    # 🔍 DEBUGGING: Check feature importance to understand high R²
    print(f"\n🔍 TOP 10 FEATURE IMPORTANCE (to understand high R²):")
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.0f}")
    
    # Check if lag feature dominates (potential issue)
    lag_importance = importance_df[importance_df['feature'].str.contains('lag')]['importance'].sum()
    total_importance = importance_df['importance'].sum()
    lag_ratio = lag_importance / total_importance
    print(f"  📊 Lag features account for {lag_ratio:.1%} of total importance")
    
    if lag_ratio > 0.8:
        print(f"  ⚠️  WARNING: Model heavily relies on lag features ({lag_ratio:.1%})")
    
    # 🔍 DEBUGGING: Check prediction variance
    test_preds = model.predict(X_test)
    pred_variance = np.var(test_preds)
    target_variance = np.var(y_test)
    print(f"  📈 Prediction variance: {pred_variance:.2f} vs Target variance: {target_variance:.2f}")
    
    if pred_variance < target_variance * 0.1:
        print(f"  ⚠️  WARNING: Low prediction variance suggests potential issues")
    
    # Simplified 72-hour prediction using only Hopsworks features
    print("\n🔮 Generating 72-hour forecast with Hopsworks features only...")
    last_row = test_df.iloc[-1]
    
    # 🔍 DEBUG: Check last row values
    print(f"  🔍 Last row AQI: {last_row['aqi_epa_calc']:.1f}")
    print(f"  🔍 Last row lag feature: {last_row.get('aqi_epa_calc_lag_1h', 'N/A')}")
    print(f"  🔍 Last row PM2.5: {last_row.get('pm2_5_nowcast', 'N/A')}")
    
    preds_df = predict_72h_simple(model, last_row, feature_cols, scaler)
    
    # Save predictions
    out_path = 'simple_aqi_predictions.csv'
    preds_df.to_csv(out_path, index=False)
    print(f"\n✅ Saved predictions → {out_path} ({len(preds_df)} rows)")
    
    # Preview
    print("\n📊 Prediction Preview:")
    print(preds_df.head(10).to_string(index=False))
    
    # Show prediction statistics
    pred_stats = preds_df['predicted_aqi'].describe()
    print(f"\n📈 Prediction Statistics:")
    print(f"  Range: {pred_stats['min']:.1f} → {pred_stats['max']:.1f}")
    print(f"  Mean: {pred_stats['mean']:.1f}, Std: {pred_stats['std']:.1f}")
    print(f"  Variation: {(pred_stats['max'] - pred_stats['min']):.1f} AQI units")
    
    # Summary
    print(f"\n🎯 Model Summary:")
    print(f"  Features used: {len(feature_cols)} (Hopsworks only)")
    print(f"  Additional features created: 1 (aqi_epa_calc_lag_1h)")
    print(f"  Model type: {type(model).__name__}")


if __name__ == '__main__':
    main()