#!/usr/bin/env python3
"""
Fix Empty Values Solution
=========================

This script provides improved feature engineering functions that handle missing/zero input data gracefully.

Author: Assistant
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pytz

def safe_feature_engineering(df):
    """
    Improved feature engineering with proper handling of missing/zero values.
    
    Args:
        df: DataFrame with pollutant data
        
    Returns:
        DataFrame with safely calculated features
    """
    df_result = df.copy()
    
    # 1. PM2.5 to PM10 Ratio (with safe division)
    def safe_pm_ratio(pm2_5, pm10):
        """Calculate PM2.5/PM10 ratio with safe division."""
        if pd.isna(pm2_5) or pd.isna(pm10) or pm10 == 0:
            return np.nan
        return pm2_5 / pm10
    
    df_result['pm2_5_pm10_ratio'] = df_result.apply(
        lambda row: safe_pm_ratio(row.get('pm2_5_nowcast'), row.get('pm10_nowcast')), 
        axis=1
    )
    
    # 2. Traffic Index (with input validation)
    def safe_traffic_index(co, no2):
        """Calculate traffic index with input validation."""
        if pd.isna(co) or pd.isna(no2) or co == 0 or no2 == 0:
            return np.nan
        return co * no2
    
    df_result['traffic_index'] = df_result.apply(
        lambda row: safe_traffic_index(row.get('co_ppm_8hr_avg'), row.get('no2_ppb')), 
        axis=1
    )
    
    # 3. Total PM (with input validation)
    def safe_total_pm(pm2_5, pm10):
        """Calculate total PM with input validation."""
        if pd.isna(pm2_5) or pd.isna(pm10):
            return np.nan
        return pm2_5 + pm10
    
    df_result['total_pm'] = df_result.apply(
        lambda row: safe_total_pm(row.get('pm2_5_nowcast'), row.get('pm10_nowcast')), 
        axis=1
    )
    
    # 4. PM Weighted (with input validation)
    def safe_pm_weighted(pm2_5, pm10):
        """Calculate weighted PM with input validation."""
        if pd.isna(pm2_5) or pd.isna(pm10):
            return np.nan
        return (pm2_5 * 0.6) + (pm10 * 0.4)
    
    df_result['pm_weighted'] = df_result.apply(
        lambda row: safe_pm_weighted(row.get('pm2_5_nowcast'), row.get('pm10_nowcast')), 
        axis=1
    )
    
    # 5. NO2 Lag (this is expected to have NaN in first row)
    df_result['no2_ppb_lag_1h'] = df_result['no2_ppb'].shift(1)
    
    return df_result

def validate_input_data(df):
    """
    Validate input data quality before feature engineering.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (is_valid, issues_list)
    """
    issues = []
    required_cols = ['pm2_5_nowcast', 'pm10_nowcast', 'co_ppm_8hr_avg', 'no2_ppb']
    
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")
            continue
            
        # Check for missing values
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            issues.append(f"{col}: {missing_count} missing values")
        
        # Check for zero values (problematic for ratios and multiplications)
        if col in ['pm10_nowcast', 'co_ppm_8hr_avg', 'no2_ppb']:
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                issues.append(f"{col}: {zero_count} zero values (may cause calculation issues)")
    
    return len(issues) == 0, issues

def improved_apply_all_feature_engineering(df):
    """
    Improved version of apply_all_feature_engineering with better error handling.
    
    This function should replace the current one in unified_aqi_hopsworks_pipeline.py
    """
    print("🔧 Starting improved feature engineering...")
    
    # Validate input data
    is_valid, issues = validate_input_data(df)
    if not is_valid:
        print("⚠️  Input data validation issues found:")
        for issue in issues:
            print(f"   - {issue}")
    
    # Apply safe feature engineering
    df_features = safe_feature_engineering(df)
    
    # Add time-based features (these should work fine)
    df_features['hour'] = df_features['datetime'].dt.hour
    df_features['is_rush_hour'] = df_features['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df_features['is_weekend'] = df_features['datetime'].dt.weekday.isin([5, 6]).astype(int)
    
    # Season calculation
    def get_season(month):
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall
    
    df_features['season'] = df_features['datetime'].dt.month.apply(get_season)
    
    # PM2.5 lag feature
    df_features['pm2_5_nowcast_lag_1h'] = df_features['pm2_5_nowcast'].shift(1)
    
    # DateTime ID (Unix timestamp)
    PKT = pytz.timezone('Asia/Karachi')
    df_features['datetime_id'] = df_features['datetime'].apply(
        lambda x: int(PKT.localize(x).timestamp()) if pd.notna(x) else np.nan
    )
    
    # Report feature engineering results
    print("✅ Feature engineering completed!")
    
    # Report missing values in engineered features
    engineered_features = ['pm2_5_pm10_ratio', 'traffic_index', 'total_pm', 'pm_weighted', 'no2_ppb_lag_1h']
    for feature in engineered_features:
        if feature in df_features.columns:
            missing_count = df_features[feature].isnull().sum()
            total_count = len(df_features)
            print(f"   📊 {feature}: {total_count - missing_count}/{total_count} valid values ({missing_count} missing)")
    
    return df_features

def demonstrate_fix():
    """Demonstrate the fix using sample data."""
    print("🔍 DEMONSTRATING EMPTY VALUES FIX")
    print("=" * 60)
    
    # Create sample data that mimics the problematic scenario
    sample_data = [
        # First few rows with missing/zero data (problematic)
        {'datetime': datetime(2024, 10, 28, 0, 0), 'pm2_5_nowcast': np.nan, 'pm10_nowcast': np.nan, 'co_ppm_8hr_avg': np.nan, 'no2_ppb': np.nan},
        {'datetime': datetime(2024, 10, 28, 1, 0), 'pm2_5_nowcast': np.nan, 'pm10_nowcast': np.nan, 'co_ppm_8hr_avg': np.nan, 'no2_ppb': np.nan},
        {'datetime': datetime(2024, 10, 28, 2, 0), 'pm2_5_nowcast': 50.0, 'pm10_nowcast': 0.0, 'co_ppm_8hr_avg': np.nan, 'no2_ppb': 15.0},  # Zero PM10
        {'datetime': datetime(2024, 10, 28, 3, 0), 'pm2_5_nowcast': 45.0, 'pm10_nowcast': 90.0, 'co_ppm_8hr_avg': 0.0, 'no2_ppb': 12.0},    # Zero CO
        {'datetime': datetime(2024, 10, 28, 4, 0), 'pm2_5_nowcast': 55.0, 'pm10_nowcast': 110.0, 'co_ppm_8hr_avg': 0.5, 'no2_ppb': 0.0},   # Zero NO2
        # Normal data
        {'datetime': datetime(2024, 10, 28, 5, 0), 'pm2_5_nowcast': 60.0, 'pm10_nowcast': 120.0, 'co_ppm_8hr_avg': 0.6, 'no2_ppb': 18.0},
        {'datetime': datetime(2024, 10, 28, 6, 0), 'pm2_5_nowcast': 52.0, 'pm10_nowcast': 105.0, 'co_ppm_8hr_avg': 0.4, 'no2_ppb': 14.0},
    ]
    
    df_sample = pd.DataFrame(sample_data)
    df_sample['aqi_epa_calc'] = 100  # Add required column
    
    print("📊 Sample Input Data:")
    print(df_sample[['datetime', 'pm2_5_nowcast', 'pm10_nowcast', 'co_ppm_8hr_avg', 'no2_ppb']].to_string())
    
    # Apply improved feature engineering
    df_result = improved_apply_all_feature_engineering(df_sample)
    
    print("\n📊 Results with Improved Feature Engineering:")
    result_cols = ['datetime', 'pm2_5_pm10_ratio', 'traffic_index', 'total_pm', 'pm_weighted', 'no2_ppb_lag_1h']
    print(df_result[result_cols].to_string())
    
    print("\n✅ Notice how the improved version handles missing/zero data gracefully!")

if __name__ == "__main__":
    demonstrate_fix()