#!/usr/bin/env python3
"""
Test script for feature engineering functions in unified_aqi_hopsworks_pipeline.py
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import zoneinfo

# Import functions from our pipeline
from unified_aqi_hopsworks_pipeline import (
    add_engineered_features,
    add_temporal_features, 
    add_lag_features,
    apply_all_feature_engineering,
    PRODUCTION_FEATURES_SCHEMA
)

PKT = zoneinfo.ZoneInfo("Asia/Karachi")

def create_sample_data():
    """Create sample data for testing feature engineering."""
    # Create sample timestamps (24 hours of data)
    timestamps = pd.date_range(
        start=datetime(2024, 1, 15, 0, 0, tzinfo=PKT),
        periods=24,
        freq='H'
    )
    
    # Create sample pollutant data
    np.random.seed(42)  # For reproducible results
    
    data = {
        'datetime': timestamps,
        'pm2_5_nowcast': np.random.uniform(20, 80, 24),
        'pm10_nowcast': np.random.uniform(40, 120, 24),
        'co_ppm_8hr_avg': np.random.uniform(1.0, 5.0, 24),
        'no2_ppb': np.random.uniform(10, 50, 24),
        'so2_ppb': np.random.uniform(5, 25, 24),
        'aqi_epa_calc': np.random.randint(50, 150, 24)
    }
    
    return pd.DataFrame(data)

def test_engineered_features():
    """Test engineered feature calculations."""
    print("🧪 Testing Engineered Features...")
    
    df = create_sample_data()
    df_eng = add_engineered_features(df)
    
    # Check if all engineered features are present
    expected_features = ['pm2_5_pm10_ratio', 'traffic_index', 'total_pm', 'pm_weighted']
    
    for feature in expected_features:
        if feature in df_eng.columns:
            print(f"  ✅ {feature}: {df_eng[feature].iloc[0]:.3f}")
        else:
            print(f"  ❌ {feature}: Missing!")
    
    # Verify calculations for first row
    row = df_eng.iloc[0]
    
    # Test PM ratio calculation
    expected_ratio = row['pm2_5_nowcast'] / (row['pm10_nowcast'] + 1e-6)
    actual_ratio = row['pm2_5_pm10_ratio']
    print(f"  📊 PM2.5/PM10 ratio: Expected {expected_ratio:.3f}, Got {actual_ratio:.3f}")
    
    # Test traffic index
    expected_traffic = (row['no2_ppb'] * 0.6) + (row['co_ppm_8hr_avg'] * 0.4)
    actual_traffic = row['traffic_index']
    print(f"  🚗 Traffic index: Expected {expected_traffic:.3f}, Got {actual_traffic:.3f}")
    
    return df_eng

def test_temporal_features():
    """Test temporal feature calculations."""
    print("\n🕐 Testing Temporal Features...")
    
    df = create_sample_data()
    df_temp = add_temporal_features(df)
    
    # Check temporal features
    expected_features = ['hour', 'is_rush_hour', 'is_weekend', 'season']
    
    for feature in expected_features:
        if feature in df_temp.columns:
            print(f"  ✅ {feature}: Sample values {df_temp[feature].head(3).tolist()}")
        else:
            print(f"  ❌ {feature}: Missing!")
    
    # Test specific calculations
    first_row = df_temp.iloc[0]
    print(f"  📅 First timestamp: {first_row['datetime']}")
    print(f"  🕐 Hour: {first_row['hour']}")
    print(f"  🚗 Rush hour: {first_row['is_rush_hour']}")
    print(f"  📅 Weekend: {first_row['is_weekend']}")
    print(f"  🌱 Season: {first_row['season']}")
    
    return df_temp

def test_lag_features():
    """Test lag feature calculations."""
    print("\n⏰ Testing Lag Features...")
    
    df = create_sample_data()
    df_lag = add_lag_features(df)
    
    # Check lag features
    expected_features = ['pm2_5_nowcast_lag_1h', 'no2_ppb_lag_1h']
    
    for feature in expected_features:
        if feature in df_lag.columns:
            non_null_count = df_lag[feature].notna().sum()
            print(f"  ✅ {feature}: {non_null_count} non-null values (expected: {len(df_lag)-1})")
        else:
            print(f"  ❌ {feature}: Missing!")
    
    # Verify lag calculation
    print(f"  📊 PM2.5 original[1]: {df_lag['pm2_5_nowcast'].iloc[1]:.3f}")
    print(f"  📊 PM2.5 lag_1h[1]: {df_lag['pm2_5_nowcast_lag_1h'].iloc[1]:.3f}")
    print(f"  📊 Should match PM2.5 original[0]: {df_lag['pm2_5_nowcast'].iloc[0]:.3f}")
    
    return df_lag

def test_full_feature_engineering():
    """Test complete feature engineering pipeline."""
    print("\n🔧 Testing Complete Feature Engineering Pipeline...")
    
    df = create_sample_data()
    df_features = apply_all_feature_engineering(df)
    
    print(f"  📊 Input shape: {df.shape}")
    print(f"  📊 Output shape: {df_features.shape}")
    print(f"  📊 Expected features: {len(PRODUCTION_FEATURES_SCHEMA)}")
    
    # Check all expected features are present
    expected_features = list(PRODUCTION_FEATURES_SCHEMA.keys())
    missing_features = []
    present_features = []
    
    for feature in expected_features:
        if feature in df_features.columns:
            present_features.append(feature)
        else:
            missing_features.append(feature)
    
    print(f"\n  ✅ Present features ({len(present_features)}):")
    for feature in present_features:
        print(f"    - {feature}")
    
    if missing_features:
        print(f"\n  ❌ Missing features ({len(missing_features)}):")
        for feature in missing_features:
            print(f"    - {feature}")
    
    # Show sample of final data
    print(f"\n  📋 Sample of engineered features:")
    print(df_features.head(3).to_string())
    
    return df_features

def main():
    """Run all feature engineering tests."""
    print("🧪 FEATURE ENGINEERING TESTS")
    print("=" * 50)
    
    try:
        # Test individual components
        test_engineered_features()
        test_temporal_features()
        test_lag_features()
        
        # Test complete pipeline
        df_final = test_full_feature_engineering()
        
        print(f"\n✅ ALL TESTS COMPLETED")
        print(f"📊 Final dataset has {len(df_final)} rows and {len(df_final.columns)} features")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)