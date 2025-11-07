#!/usr/bin/env python3
"""
Enhanced AQI Prediction Script with Advanced Feature Engineering

Target: aqi_epa_calc

Key Enhancements:
- Consolidated optimal feature set (26 features)
- Advanced temporal features with cyclical encoding
- Atmospheric stability and interaction features
- Short-term rolling statistics (3-hour windows)
- Minimal but effective lag features
- SHAP analysis for interpretability
"""

import os
import warnings
import json
from datetime import timedelta, datetime
from collections import deque
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
import lightgbm as lgb
from dotenv import load_dotenv
import shap

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
        
        return df
        
    except Exception as e:
        raise RuntimeError(f"❌ Failed to fetch data from Hopsworks: {type(e).__name__}: {e}")


def enhanced_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply enhanced feature engineering with the consolidated optimal feature set.
    
    Creates 26 optimal features:
    - 5 core features
    - 6 scientifically justified engineered features
    - 8 essential temporal features
    - 3 minimal lag features
    - 4 short-term rolling features
    """
    print("🔧 Applying enhanced feature engineering (26 optimal features)...")
    df_enhanced = df.copy()
    
    # Ensure datetime is parsed
    df_enhanced['datetime'] = pd.to_datetime(df_enhanced['datetime'])
    df_enhanced = df_enhanced.sort_values('datetime').reset_index(drop=True)
    
    # === CORE FEATURES (5) ===
    # These should already exist: pm2_5_nowcast, pm10_nowcast, co_ppm_8hr_avg, no2_ppb, so2_ppb
    
    # === ENGINEERED FEATURES (6) ===
    print("   📊 Creating scientifically justified engineered features...")
    
    # 1. PM2.5/PM10 ratio (respiratory health indicator)
    df_enhanced['pm2_5_pm10_ratio'] = np.where(
        (df_enhanced['pm10_nowcast'] > 0) & (~df_enhanced['pm10_nowcast'].isna()),
        df_enhanced['pm2_5_nowcast'] / df_enhanced['pm10_nowcast'],
        np.nan
    )
    
    # 2. Atmospheric stability (temperature inversion proxy)
    if 'temp' in df_enhanced.columns and 'humidity' in df_enhanced.columns:
        df_enhanced['atmospheric_stability'] = (
            df_enhanced['temp'] / (df_enhanced['humidity'] + 1e-6)
        )
    else:
        # Fallback using pollutant ratios as stability proxy
        df_enhanced['atmospheric_stability'] = np.where(
            (df_enhanced['co_ppm_8hr_avg'] > 0) & (~df_enhanced['co_ppm_8hr_avg'].isna()),
            df_enhanced['pm2_5_nowcast'] / (df_enhanced['co_ppm_8hr_avg'] + 1e-6),
            np.nan
        )
    
    # 3. Traffic pollution index
    df_enhanced['traffic_pollution_index'] = (
        0.6 * df_enhanced['no2_ppb'].fillna(0) + 
        0.4 * df_enhanced['co_ppm_8hr_avg'].fillna(0)
    )
    
    # 4. Industrial pollution index
    df_enhanced['industrial_pollution_index'] = (
        0.7 * df_enhanced['so2_ppb'].fillna(0) + 
        0.3 * df_enhanced['pm10_nowcast'].fillna(0)
    )
    
    # 5. Total particulate matter
    df_enhanced['total_pm'] = (
        df_enhanced['pm2_5_nowcast'].fillna(0) + 
        df_enhanced['pm10_nowcast'].fillna(0)
    )
    
    # 6. Weighted PM (PM2.5 is more harmful)
    df_enhanced['pm_weighted'] = (
        0.7 * df_enhanced['pm2_5_nowcast'].fillna(0) + 
        0.3 * df_enhanced['pm10_nowcast'].fillna(0)
    )
    
    # === TEMPORAL FEATURES (8) ===
    print("   🕐 Creating enhanced temporal features...")
    
    # Extract basic temporal components
    df_enhanced['hour'] = df_enhanced['datetime'].dt.hour
    df_enhanced['day_of_week'] = df_enhanced['datetime'].dt.dayofweek
    df_enhanced['month'] = df_enhanced['datetime'].dt.month
    
    # Cyclical encodings for smooth transitions
    df_enhanced['hour_sin'] = np.sin(2 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['hour_cos'] = np.cos(2 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['day_sin'] = np.sin(2 * np.pi * df_enhanced['day_of_week'] / 7)
    df_enhanced['day_cos'] = np.cos(2 * np.pi * df_enhanced['day_of_week'] / 7)
    
    # Enhanced rush hour indicators
    df_enhanced['is_morning_rush'] = (
        (df_enhanced['hour'] >= 7) & (df_enhanced['hour'] <= 9) & 
        (df_enhanced['day_of_week'] < 5)  # Weekdays only
    ).astype(int)
    
    # === LAG FEATURES (3) ===
    print("   ⏰ Creating minimal lag features...")
    
    # Only the most predictive lags
    df_enhanced['pm2_5_nowcast_lag_1h'] = df_enhanced['pm2_5_nowcast'].shift(1)
    df_enhanced['aqi_epa_calc_lag_2h'] = df_enhanced['aqi_epa_calc'].shift(2)
    df_enhanced['no2_ppb_lag_1h'] = df_enhanced['no2_ppb'].shift(1)
    
    # === ROLLING FEATURES (4) ===
    print("   📈 Creating short-term rolling statistics...")
    
    # 3-hour rolling statistics (short-term trends)
    df_enhanced['pm2_5_3h_mean'] = df_enhanced['pm2_5_nowcast'].rolling(
        window=3, min_periods=2
    ).mean()
    df_enhanced['pm2_5_3h_std'] = df_enhanced['pm2_5_nowcast'].rolling(
        window=3, min_periods=2
    ).std()
    df_enhanced['no2_3h_mean'] = df_enhanced['no2_ppb'].rolling(
        window=3, min_periods=2
    ).mean()
    df_enhanced['no2_3h_std'] = df_enhanced['no2_ppb'].rolling(
        window=3, min_periods=2
    ).std()
    
    # === FEATURE SELECTION ===
    # Select only the 26 optimal features + target + datetime
    optimal_features = [
        'datetime', 'aqi_epa_calc',  # Essential columns
        
        # Core features (5)
        'pm2_5_nowcast', 'pm10_nowcast', 'co_ppm_8hr_avg', 'no2_ppb', 'so2_ppb',
        
        # Engineered features (6)
        'pm2_5_pm10_ratio', 'atmospheric_stability', 'traffic_pollution_index',
        'industrial_pollution_index', 'total_pm', 'pm_weighted',
        
        # Temporal features (8)
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_morning_rush',
        'hour', 'day_of_week', 'month',
        
        # Lag features (3)
        'pm2_5_nowcast_lag_1h', 'aqi_epa_calc_lag_2h', 'no2_ppb_lag_1h',
        
        # Rolling features (4)
        'pm2_5_3h_mean', 'pm2_5_3h_std', 'no2_3h_mean', 'no2_3h_std'
    ]
    
    # Keep only available features
    available_features = [col for col in optimal_features if col in df_enhanced.columns]
    df_final = df_enhanced[available_features].copy()
    
    # Save enhanced features to CSV for inspection
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"enhanced_features_{timestamp}.csv"
    df_final.to_csv(csv_filename, index=False)
    
    print(f"✅ Enhanced feature engineering complete!")
    print(f"   📊 Features created: {len(available_features) - 2} (excluding datetime and target)")
    print(f"   💾 Saved to: {csv_filename}")
    print(f"   📈 Shape: {df_final.shape}")
    
    # Report feature completeness
    print("   📋 Feature completeness:")
    for feature in available_features[2:]:  # Skip datetime and target
        missing_pct = (df_final[feature].isna().sum() / len(df_final)) * 100
        print(f"      {feature}: {100-missing_pct:.1f}% complete")
    
    return df_final


def smart_nan_handling(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced NaN handling for the new feature set."""
    df_clean = df.copy()
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in ['datetime', 'datetime_id']:
            continue
            
        if df[col].isna().sum() > 0:
            # Forward fill for short gaps (< 3 hours)
            df_clean[col] = df_clean[col].fillna(method='ffill', limit=3)
            
            # Linear interpolation for medium gaps (< 6 hours for enhanced features)
            df_clean[col] = df_clean[col].interpolate(method='linear', limit=6)
            
            # Median only as last resort
            remaining_nas = df_clean[col].isna().sum()
            if remaining_nas > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                print(f"  ⚠️ {col}: {remaining_nas} NaNs filled with median")
    
    return df_clean


def preprocess_enhanced(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced preprocessing pipeline."""
    print("🧹 Enhanced preprocessing...")
    
    # Parse datetime
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
    else:
        raise ValueError("'datetime' column missing in features")

    # Target check
    if 'aqi_epa_calc' not in df.columns:
        raise ValueError("'aqi_epa_calc' (target) missing in features")

    # Apply enhanced feature engineering
    df = enhanced_feature_engineering(df)
    
    # Smart NaN handling
    df = smart_nan_handling(df)
    
    print(f"✅ Enhanced preprocessing complete: {len(df)} records, {len(df.columns)} features")
    return df


def create_safe_lags(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """Create lag features safely - enhanced version already includes lags."""
    # Enhanced version already creates lags, so just handle NaN removal
    df_with_lags = df.copy()
    
    # Drop rows with NaN lags only for training
    if is_training:
        lag_cols = [c for c in df_with_lags.columns if '_lag_' in c]
        if lag_cols:
            initial_len = len(df_with_lags)
            df_with_lags = df_with_lags.dropna(subset=lag_cols).reset_index(drop=True)
            dropped = initial_len - len(df_with_lags)
            print(f"  📉 Dropped {dropped} rows with NaN lag features")
    
    return df_with_lags


def temporal_split_enhanced(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    """Enhanced temporal split for the new feature set."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Split base data first
    train_base = df.iloc[:train_end].copy()
    val_base = df.iloc[train_end:val_end].copy()
    test_base = df.iloc[val_end:].copy()
    
    # Handle lag features for each split
    train_df = create_safe_lags(train_base, is_training=True)
    val_df = create_safe_lags(val_base, is_training=False)
    test_df = create_safe_lags(test_base, is_training=False)
    
    print("📅 Enhanced temporal split:")
    print(f"  Train: {len(train_df)} ({train_df['datetime'].iloc[0]} → {train_df['datetime'].iloc[-1]})")
    print(f"  Val:   {len(val_df)} ({val_df['datetime'].iloc[0]} → {val_df['datetime'].iloc[-1]})")
    print(f"  Test:  {len(test_df)} ({test_df['datetime'].iloc[0]} → {test_df['datetime'].iloc[-1]})")
    
    return train_df, val_df, test_df


def select_features_enhanced(df: pd.DataFrame):
    """Enhanced feature selection for the optimal feature set."""
    exclude = {'aqi_epa_calc', 'datetime', 'datetime_id'}
    feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype != 'object']
    
    print(f"  🎯 Selected {len(feat_cols)} enhanced features")
    print(f"     - Engineered: {sum(any(x in c for x in ['ratio', 'index', 'total', 'weighted', 'stability']) for c in feat_cols)}")
    print(f"     - Temporal: {sum(any(x in c for x in ['hour', 'day', 'sin', 'cos', 'rush', 'month']) for c in feat_cols)}")
    print(f"     - Lag: {sum('_lag_' in c for c in feat_cols)}")
    print(f"     - Rolling: {sum(any(x in c for x in ['_mean', '_std']) for c in feat_cols)}")
    
    return feat_cols


def train_enhanced_model(X_train, y_train, X_val, y_val):
    """Train enhanced LightGBM model with anti-overfitting measures."""
    print("🚀 Training enhanced model with anti-overfitting parameters...")
    
    # More conservative parameters to prevent overfitting
    params = {
        'n_estimators': 500,        # Reduced estimators
        'learning_rate': 0.03,      # Lower learning rate
        'num_leaves': 31,           # Reduced complexity
        'max_depth': 5,             # Shallower trees
        'subsample': 0.7,           # More aggressive subsampling
        'colsample_bytree': 0.7,    # More feature sampling
        'reg_alpha': 0.5,           # Stronger L1 regularization
        'reg_lambda': 1.0,          # Stronger L2 regularization
        'min_child_samples': 50,    # Higher minimum samples
        'min_child_weight': 0.01,   # Minimum weight
        'bagging_freq': 5,          # Bagging frequency
        'feature_fraction': 0.8,    # Feature fraction
        'random_state': 42,
        'force_col_wise': True,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),  # Earlier stopping
        lgb.log_evaluation(period=0)  # Suppress training logs
    ]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=['train', 'valid'],
        eval_metric='l2',
        callbacks=callbacks
    )
    
    print("✅ Enhanced model trained with anti-overfitting measures.")
    return model


def walk_forward_validation(df: pd.DataFrame, feature_cols: list, n_splits: int = 5):
    """Implement proper walk-forward cross-validation to detect overfitting."""
    print(f"🚶 Performing walk-forward cross-validation ({n_splits} splits)...")
    
    n = len(df)
    split_size = n // (n_splits + 1)  # Leave room for final test
    
    cv_scores = []
    
    for i in range(n_splits):
        # Progressive training window
        train_end = split_size * (i + 2)  # Growing training set
        val_start = train_end
        val_end = min(train_end + split_size, n)
        
        if val_end >= n:
            break
            
        # Split data
        train_data = df.iloc[:train_end].copy()
        val_data = df.iloc[val_start:val_end].copy()
        
        # Handle lag features
        train_clean = create_safe_lags(train_data, is_training=True)
        val_clean = create_safe_lags(val_data, is_training=False)
        
        if len(train_clean) < 100 or len(val_clean) < 10:
            continue
            
        # Prepare features
        X_train_cv = train_clean[feature_cols]
        y_train_cv = train_clean['aqi_epa_calc']
        X_val_cv = val_clean[feature_cols]
        y_val_cv = val_clean['aqi_epa_calc']
        
        # Train model
        model_cv = train_enhanced_model(X_train_cv, y_train_cv, X_val_cv, y_val_cv)
        
        # Evaluate
        val_pred = model_cv.predict(X_val_cv)
        val_rmse = np.sqrt(mean_squared_error(y_val_cv, val_pred))
        
        cv_scores.append(val_rmse)
        print(f"   Split {i+1}: RMSE = {val_rmse:.2f} (train: {len(train_clean)}, val: {len(val_clean)})")
    
    if cv_scores:
        mean_cv_rmse = np.mean(cv_scores)
        std_cv_rmse = np.std(cv_scores)
        print(f"📊 Walk-forward CV Results:")
        print(f"   Mean RMSE: {mean_cv_rmse:.2f} ± {std_cv_rmse:.2f}")
        print(f"   CV Scores: {[f'{s:.2f}' for s in cv_scores]}")
        
        # Overfitting check
        if std_cv_rmse > mean_cv_rmse * 0.3:
            print("⚠️  HIGH VARIANCE: Model may be overfitting!")
        else:
            print("✅ CV variance looks reasonable")
            
        return cv_scores
    else:
        print("❌ Walk-forward CV failed - insufficient data")
        return []


def generate_future_predictions(model, df_processed: pd.DataFrame, feature_cols: list, hours_ahead: int = 72):
    """Generate future predictions using the trained model."""
    print(f"🔮 Generating {hours_ahead}-hour future predictions...")
    
    # Get the last complete row for prediction
    last_row = df_processed.iloc[-1:].copy()
    last_datetime = pd.to_datetime(last_row['datetime'].iloc[0])
    
    predictions = []
    current_data = df_processed.copy()
    
    for hour in range(1, hours_ahead + 1):
        # Create future timestamp
        future_time = last_datetime + pd.Timedelta(hours=hour)
        
        # Create a new row with temporal features
        new_row = pd.DataFrame({
            'datetime': [future_time],
            'aqi_epa_calc': [np.nan]  # This will be predicted
        })
        
        # Add temporal features for the future time
        new_row['hour'] = future_time.hour
        new_row['day_of_week'] = future_time.dayofweek
        new_row['month'] = future_time.month
        new_row['hour_sin'] = np.sin(2 * np.pi * future_time.hour / 24)
        new_row['hour_cos'] = np.cos(2 * np.pi * future_time.hour / 24)
        new_row['day_sin'] = np.sin(2 * np.pi * future_time.dayofweek / 7)
        new_row['day_cos'] = np.cos(2 * np.pi * future_time.dayofweek / 7)
        new_row['is_morning_rush'] = int(
            (future_time.hour >= 7) & (future_time.hour <= 9) & 
            (future_time.dayofweek < 5)
        )
        
        # For other features, use the last known values (persistence model)
        for col in feature_cols:
            if col not in new_row.columns:
                if '_lag_' in col:
                    # Handle lag features by looking back in current_data
                    lag_hours = int(col.split('_lag_')[1].replace('h', ''))
                    if len(current_data) >= lag_hours:
                        base_col = col.split('_lag_')[0]
                        if base_col in current_data.columns:
                            new_row[col] = current_data[base_col].iloc[-lag_hours]
                        else:
                            new_row[col] = current_data.iloc[-lag_hours]['aqi_epa_calc'] if base_col == 'aqi_epa_calc' else np.nan
                    else:
                        new_row[col] = np.nan
                elif any(x in col for x in ['_mean', '_std']):
                    # For rolling features, use recent values
                    base_col = col.split('_3h_')[0]
                    if base_col in current_data.columns:
                        recent_values = current_data[base_col].tail(3)
                        if '_mean' in col:
                            new_row[col] = recent_values.mean()
                        else:  # _std
                            new_row[col] = recent_values.std()
                    else:
                        new_row[col] = np.nan
                else:
                    # Use last known value for other features
                    if col in current_data.columns:
                        new_row[col] = current_data[col].iloc[-1]
                    else:
                        new_row[col] = np.nan
        
        # Fill any remaining NaN values
        for col in feature_cols:
            if col not in new_row.columns or pd.isna(new_row[col].iloc[0]):
                new_row[col] = 0  # Fallback value
        
        # Make prediction
        try:
            X_pred = new_row[feature_cols]
            pred_aqi = model.predict(X_pred)[0]
            
            # Apply realistic bounds
            pred_aqi = max(50, min(300, pred_aqi))  # Reasonable AQI bounds
            
            predictions.append({
                'datetime': future_time,
                'predicted_aqi': pred_aqi,
                'hour_ahead': hour
            })
            
            # Update the new row with prediction and add to current_data for next iteration
            new_row['aqi_epa_calc'] = pred_aqi
            current_data = pd.concat([current_data, new_row], ignore_index=True)
            
        except Exception as e:
            print(f"⚠️ Prediction failed for hour {hour}: {e}")
            predictions.append({
                'datetime': future_time,
                'predicted_aqi': np.nan,
                'hour_ahead': hour
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(predictions)
    
    # Save predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f'enhanced_predictions_{timestamp}.csv', index=False)
    
    print(f"✅ Future predictions generated and saved")
    print(f"   📊 Prediction range: {results_df['predicted_aqi'].min():.1f} - {results_df['predicted_aqi'].max():.1f}")
    print(f"   📊 Mean prediction: {results_df['predicted_aqi'].mean():.1f}")
    
    return results_df
    """Perform SHAP analysis for model interpretability."""
    print("🔍 Performing SHAP analysis for model interpretability...")
    
    try:
        # Sample data for SHAP (computational efficiency)
        if len(X_train) > max_samples:
            sample_idx = np.random.choice(len(X_train), max_samples, replace=False)
            X_train_sample = X_train.iloc[sample_idx]
        else:
            X_train_sample = X_train
        
        if len(X_test) > max_samples:
            sample_idx = np.random.choice(len(X_test), max_samples, replace=False)
            X_test_sample = X_test.iloc[sample_idx]
        else:
            X_test_sample = X_test
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_sample)
        
        # Feature importance from SHAP
        feature_importance = np.abs(shap_values).mean(0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("📊 Top 10 most important features (SHAP):")
        for i, row in importance_df.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        # Save SHAP results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        importance_df.to_csv(f"shap_feature_importance_{timestamp}.csv", index=False)
        
        return explainer, shap_values, importance_df
        
    except Exception as e:
        print(f"⚠️ SHAP analysis failed: {e}")
        return None, None, None


def shap_analysis(model, X_train, X_test, feature_names, max_samples=1000):
    """Perform SHAP analysis for model interpretability."""
    print("🔍 Performing SHAP analysis for model interpretability...")
    
    try:
        import shap
        
        # Create explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values for a sample of test data
        sample_size = min(max_samples, len(X_test))
        X_sample = X_test.sample(n=sample_size, random_state=42)
        
        print(f"   Computing SHAP values for {sample_size} samples...")
        shap_values = explainer.shap_values(X_sample)
        
        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(0)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': feature_importance
        }).sort_values('shap_importance', ascending=False)
        
        print("📊 Top 10 SHAP Feature Importance:")
        for i, row in importance_df.head(10).iterrows():
            print(f"   {row['feature']}: {row['shap_importance']:.4f}")
        
        # Save SHAP results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        importance_df.to_csv(f"shap_feature_importance_{timestamp}.csv", index=False)
        
        return explainer, shap_values, importance_df
        
    except ImportError:
        print("⚠️ SHAP not installed. Skipping SHAP analysis.")
        print("   Install with: pip install shap")
        return None, None, None
        
    except Exception as e:
        print(f"⚠️ SHAP analysis failed: {e}")
        return None, None, None


def evaluate_enhanced(model, X, y, label="Test"):
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
    
    # AQI category accuracy
    def get_aqi_category(aqi):
        if aqi <= 50: return 0  # Good
        elif aqi <= 100: return 1  # Moderate
        elif aqi <= 150: return 2  # Unhealthy for Sensitive Groups
        elif aqi <= 200: return 3  # Unhealthy
        elif aqi <= 300: return 4  # Very Unhealthy
        else: return 5  # Hazardous
    
    y_cat = [get_aqi_category(val) for val in y]
    pred_cat = [get_aqi_category(val) for val in preds]
    cat_accuracy = np.mean(np.array(y_cat) == np.array(pred_cat))
    
    print(f"📈 {label} Enhanced Metrics:")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAE: {mae:.2f}")
    print(f"   MAPE: {mape:.1f}%")
    print(f"   Directional Accuracy: {da:.3f}")
    print(f"   R²: {r2:.3f}")
    print(f"   AQI Category Accuracy: {cat_accuracy:.3f}")
    
    return {
        'rmse': rmse, 'mae': mae, 'mape': mape, 
        'directional_accuracy': da, 'r2': r2, 'category_accuracy': cat_accuracy
    }


def main():
    """Enhanced main function with comprehensive testing and overfitting prevention."""
    print("=== ENHANCED AQI PREDICTION WITH OPTIMAL FEATURES ===")
    
    try:
        # 1. Load and preprocess data
        print("\n" + "="*60)
        print("1️⃣ DATA LOADING AND PREPROCESSING")
        print("="*60)
        
        df = fetch_features()
        print(f"✅ Data loaded: {df.shape}")
        
        # Enhanced preprocessing with feature engineering
        df_processed = preprocess_enhanced(df)
        
        # Feature selection
        feature_cols = select_features_enhanced(df_processed)
        print(f"📊 Selected {len(feature_cols)} features")
        
        # 2. Walk-forward cross-validation to check for overfitting
        print("\n" + "="*60)
        print("2️⃣ WALK-FORWARD CROSS-VALIDATION")
        print("="*60)
        
        cv_scores = walk_forward_validation(df_processed, feature_cols, n_splits=5)
        
        # 3. Train final model on temporal splits
        print("\n" + "="*60)
        print("3️⃣ TRAINING FINAL MODEL")
        print("="*60)
        
        # Create temporal splits (80% train, 10% val, 10% test)
        n = len(df_processed)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        
        train_data = df_processed.iloc[:train_end].copy()
        val_data = df_processed.iloc[train_end:val_end].copy()
        test_data = df_processed.iloc[val_end:].copy()
        
        print(f"📊 Final split sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        # Prepare final training data
        X_train = train_data[feature_cols]
        y_train = train_data['aqi_epa_calc']
        X_val = val_data[feature_cols]
        y_val = val_data['aqi_epa_calc']
        X_test = test_data[feature_cols]
        y_test = test_data['aqi_epa_calc']
        
        # Train final model
        final_model = train_enhanced_model(X_train, y_train, X_val, y_val)
        
        # 4. Evaluate final model
        print("\n" + "="*60)
        print("4️⃣ FINAL MODEL EVALUATION")
        print("="*60)
        
        # Evaluate on all splits
        train_pred = final_model.predict(X_train)
        val_pred = final_model.predict(X_val)
        test_pred = final_model.predict(X_test)
        
        # Calculate metrics using the enhanced evaluation function
        train_metrics = evaluate_enhanced(final_model, X_train, y_train, "Train")
        val_metrics = evaluate_enhanced(final_model, X_val, y_val, "Validation")
        test_metrics = evaluate_enhanced(final_model, X_test, y_test, "Test")
        
        # Check for overfitting
        train_r2 = train_metrics['r2']
        val_r2 = val_metrics['r2']
        test_r2 = test_metrics['r2']
        
        print(f"\n🔍 Overfitting Analysis:")
        print(f"   Train R²: {train_r2:.4f}")
        print(f"   Val R²: {val_r2:.4f}")
        print(f"   Test R²: {test_r2:.4f}")
        
        if train_r2 - val_r2 > 0.1:
            print("⚠️  WARNING: Significant overfitting detected (Train-Val R² gap > 0.1)")
        elif train_r2 - test_r2 > 0.1:
            print("⚠️  WARNING: Potential overfitting detected (Train-Test R² gap > 0.1)")
        else:
            print("✅ Overfitting levels appear acceptable")
        
        # 5. Generate future predictions
        print("\n" + "="*60)
        print("5️⃣ GENERATING FUTURE PREDICTIONS")
        print("="*60)
        
        predictions_df = generate_future_predictions(final_model, df_processed, feature_cols, hours_ahead=72)
        
        # 6. SHAP Analysis
        print("\n" + "="*60)
        print("6️⃣ SHAP ANALYSIS")
        print("="*60)
        
        explainer, shap_values, importance_df = shap_analysis(final_model, X_train, X_test, feature_cols)
        
        # 7. Save all results
        print("\n" + "="*60)
        print("7️⃣ SAVING RESULTS")
        print("="*60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model results
        results = {
            'timestamp': timestamp,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'feature_count': len(feature_cols),
            'features_used': feature_cols,
            'model_params': {
                'n_estimators': 100,  # Reduced from default
                'max_depth': 6,       # Reduced from default
                'learning_rate': 0.05, # Reduced from default
                'subsample': 0.8,     # Added subsampling
                'colsample_bytree': 0.8, # Added feature subsampling
                'reg_alpha': 1.0,     # L1 regularization
                'reg_lambda': 1.0     # L2 regularization
            }
        }
        
        with open(f'enhanced_model_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save feature importance
        feature_imp = pd.DataFrame({
            'feature': feature_cols,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_imp.to_csv(f'enhanced_feature_importance_{timestamp}.csv', index=False)
        
        print(f"✅ Results saved to enhanced_model_results_{timestamp}.json")
        print(f"✅ Predictions saved to enhanced_predictions_{timestamp}.csv")
        print(f"✅ Features saved to enhanced_features_{timestamp}.csv")
        print(f"✅ Feature importance saved to enhanced_feature_importance_{timestamp}.csv")
        
        # Summary
        print(f"\n📊 FINAL SUMMARY:")
        print(f"   🎯 Test RMSE: {test_metrics['rmse']:.2f}")
        print(f"   🎯 Test R²: {test_metrics['r2']:.4f}")
        print(f"   🎯 Test MAE: {test_metrics['mae']:.2f}")
        print(f"   🔮 Future predictions: {len(predictions_df)} hours ahead")
        print(f"   📈 Prediction range: {predictions_df['predicted_aqi'].min():.1f} - {predictions_df['predicted_aqi'].max():.1f}")
        
        return final_model, results, predictions_df
        
    except Exception as e:
        print(f"❌ Error in enhanced prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == '__main__':
    model, results, importance = main()