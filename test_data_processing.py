#!/usr/bin/env python3
"""
Test script for data processing and quality validation in unified_aqi_hopsworks_pipeline.py
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import zoneinfo

# Import functions from our pipeline
from unified_aqi_hopsworks_pipeline import (
    fetch_aqi_data_with_retry,
    json_to_dataframe,
    validate_data_quality,
    LAT, LON, PKT
)

def test_api_fetch():
    """Test API data fetching with a small time window."""
    print("🌐 Testing API Data Fetching...")
    
    # Test with last 2 hours
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(hours=2)
    
    print(f"  📅 Fetching data from {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
    
    try:
        json_data = fetch_aqi_data_with_retry(LAT, LON, start_date, end_date)
        
        if json_data:
            print(f"  ✅ API fetch successful")
            print(f"  📊 Records received: {len(json_data.get('list', []))}")
            
            # Show sample of raw data
            if json_data.get('list'):
                sample = json_data['list'][0]
                print(f"  📋 Sample record keys: {list(sample.keys())}")
                print(f"  📋 Sample components: {list(sample.get('components', {}).keys())}")
            
            return json_data
        else:
            print("  ❌ API fetch failed")
            return None
            
    except Exception as e:
        print(f"  ❌ API fetch error: {str(e)}")
        return None

def test_data_processing(json_data):
    """Test JSON to DataFrame conversion and processing."""
    print("\n🔄 Testing Data Processing...")
    
    if not json_data:
        print("  ⚠️ No JSON data to process")
        return None
    
    try:
        df = json_to_dataframe(json_data)
        
        if df is not None and not df.empty:
            print(f"  ✅ DataFrame created successfully")
            print(f"  📊 Shape: {df.shape}")
            print(f"  📊 Columns: {list(df.columns)}")
            
            # Check timezone handling
            if 'datetime' in df.columns:
                sample_dt = df['datetime'].iloc[0]
                print(f"  🕐 Sample datetime: {sample_dt}")
                print(f"  🌍 Timezone: {sample_dt.tzinfo}")
                
                # Verify it's in Pakistan timezone
                if sample_dt.tzinfo == PKT:
                    print("  ✅ Timezone correctly set to Pakistan Standard Time")
                else:
                    print("  ⚠️ Timezone not set to Pakistan Standard Time")
            
            # Check for key columns
            expected_cols = ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'aqi_epa_calc']
            missing_cols = [col for col in expected_cols if col not in df.columns]
            
            if missing_cols:
                print(f"  ⚠️ Missing columns: {missing_cols}")
            else:
                print("  ✅ All expected columns present")
            
            # Show data sample
            print(f"\n  📋 Data sample:")
            print(df.head(2).to_string())
            
            return df
        else:
            print("  ❌ DataFrame creation failed or empty")
            return None
            
    except Exception as e:
        print(f"  ❌ Data processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_test_data_for_quality_checks():
    """Create test data with various quality issues."""
    print("\n🧪 Creating Test Data for Quality Validation...")
    
    # Create data with different quality issues
    timestamps = pd.date_range(
        start=datetime(2024, 1, 15, 0, 0, tzinfo=PKT),
        periods=10,
        freq='H'
    )
    
    # Good data
    good_data = {
        'datetime': timestamps,
        'pm2_5_nowcast': [25.5, 30.2, 28.1, 32.4, 29.8, 31.2, 27.9, 26.5, 30.1, 28.7],
        'pm10_nowcast': [45.2, 52.1, 48.3, 55.7, 51.2, 53.8, 47.6, 46.1, 52.3, 49.4],
        'co_ppm_8hr_avg': [2.1, 2.3, 2.0, 2.5, 2.2, 2.4, 2.1, 2.0, 2.3, 2.2],
        'no2_ppb': [15.2, 18.1, 16.5, 19.3, 17.2, 18.7, 16.1, 15.8, 18.2, 17.0],
        'so2_ppb': [8.1, 9.2, 8.5, 9.8, 8.9, 9.5, 8.3, 8.0, 9.1, 8.7],
        'aqi_epa_calc': [85, 92, 88, 95, 90, 93, 87, 86, 91, 89]
    }
    
    # Data with missing values (40% missing - should fail)
    missing_data = good_data.copy()
    missing_data['pm2_5_nowcast'] = [25.5, np.nan, np.nan, np.nan, np.nan, 31.2, 27.9, 26.5, 30.1, 28.7]
    
    # Data with acceptable missing values (20% missing - should pass)
    acceptable_missing_data = good_data.copy()
    acceptable_missing_data['pm2_5_nowcast'] = [25.5, 30.2, np.nan, np.nan, 29.8, 31.2, 27.9, 26.5, 30.1, 28.7]
    
    # Data with missing target (60% missing - should fail)
    missing_target_data = good_data.copy()
    missing_target_data['aqi_epa_calc'] = [85, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 86, 91, 89]
    
    return {
        'good': pd.DataFrame(good_data),
        'high_missing': pd.DataFrame(missing_data),
        'acceptable_missing': pd.DataFrame(acceptable_missing_data),
        'missing_target': pd.DataFrame(missing_target_data)
    }

def test_data_quality_validation():
    """Test data quality validation function."""
    print("\n🔍 Testing Data Quality Validation...")
    
    test_datasets = create_test_data_for_quality_checks()
    
    for name, df in test_datasets.items():
        print(f"\n  📊 Testing dataset: {name}")
        
        is_valid, message = validate_data_quality(df)
        
        if name == 'good' or name == 'acceptable_missing':
            expected_result = True
        else:
            expected_result = False
        
        if is_valid == expected_result:
            print(f"    ✅ Validation result: {is_valid} (Expected: {expected_result})")
            print(f"    📝 Message: {message}")
        else:
            print(f"    ❌ Validation result: {is_valid} (Expected: {expected_result})")
            print(f"    📝 Message: {message}")

def test_timezone_handling():
    """Test timezone handling in data processing."""
    print("\n🌍 Testing Timezone Handling...")
    
    # Create sample data with different timezone scenarios
    utc_time = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
    pkt_time = utc_time.astimezone(PKT)
    
    print(f"  🕐 UTC time: {utc_time}")
    print(f"  🕐 PKT time: {pkt_time}")
    print(f"  🕐 PKT offset: {pkt_time.strftime('%z')}")
    
    # Test DataFrame with timezone-aware datetime
    df_test = pd.DataFrame({
        'datetime': [pkt_time],
        'pm2_5_nowcast': [25.5],
        'aqi_epa_calc': [85]
    })
    
    # Check if timezone is preserved
    dt_col = df_test['datetime'].iloc[0]
    print(f"  📊 DataFrame datetime: {dt_col}")
    print(f"  📊 DataFrame timezone: {dt_col.tzinfo}")
    
    if dt_col.tzinfo == PKT:
        print("  ✅ Timezone correctly preserved in DataFrame")
    else:
        print("  ⚠️ Timezone not correctly preserved")
    
    return df_test

def main():
    """Run all data processing and quality tests."""
    print("🧪 DATA PROCESSING & QUALITY TESTS")
    print("=" * 50)
    
    try:
        # Test API fetching
        json_data = test_api_fetch()
        
        # Test data processing
        df = test_data_processing(json_data)
        
        # Test data quality validation
        test_data_quality_validation()
        
        # Test timezone handling
        test_timezone_handling()
        
        print(f"\n✅ ALL DATA PROCESSING TESTS COMPLETED")
        
        if df is not None:
            print(f"📊 Successfully processed {len(df)} records from API")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)