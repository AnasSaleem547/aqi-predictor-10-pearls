#!/usr/bin/env python3
"""
Analyze retrieved Karachi AQI features for data quality issues
"""

import pandas as pd
import numpy as np

def analyze_retrieved_data():
    """Analyze the retrieved CSV data for issues and anomalies."""
    
    # Load the retrieved data
    df = pd.read_csv('retrieved_karachi_aqi_features.csv')
    
    print('=== DATA OVERVIEW ===')
    print(f'Shape: {df.shape}')
    print(f'Date range: {df["datetime"].min()} to {df["datetime"].max()}')
    print()
    
    print('=== COLUMN ANALYSIS ===')
    for col in df.columns:
        if col != 'datetime':
            print(f'{col}: {df[col].dtype}, null: {df[col].isnull().sum()}, unique: {df[col].nunique()}')
    print()
    
    print('=== SUSPICIOUS VALUES ===')
    # Check for negative values in pollutants
    pollutant_cols = ['pm2_5_nowcast', 'pm10_nowcast', 'co_ppm_8hr_avg', 'no2_ppb', 'so2_ppb']
    for col in pollutant_cols:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            zero_count = (df[col] == 0).sum()
            if negative_count > 0 or zero_count > 10:
                print(f'{col}: {negative_count} negative values, {zero_count} zero values')
    
    print()
    print('=== AQI ANALYSIS ===')
    if 'aqi_epa_calc' in df.columns:
        print(f'AQI range: {df["aqi_epa_calc"].min()} to {df["aqi_epa_calc"].max()}')
        print(f'AQI mean: {df["aqi_epa_calc"].mean():.1f}')
        print(f'AQI > 300 count: {(df["aqi_epa_calc"] > 300).sum()}')
        print(f'AQI null count: {df["aqi_epa_calc"].isnull().sum()}')
    
    print()
    print('=== FEATURE ENGINEERING ISSUES ===')
    # Check for missing engineered features
    if 'pm2_5_pm10_ratio' in df.columns:
        missing_ratio = df['pm2_5_pm10_ratio'].isnull().sum()
        print(f'Missing PM2.5/PM10 ratio: {missing_ratio}')
        
        # Check for unrealistic ratios
        if missing_ratio < len(df):
            ratio_stats = df['pm2_5_pm10_ratio'].describe()
            print(f'PM2.5/PM10 ratio stats: min={ratio_stats["min"]:.3f}, max={ratio_stats["max"]:.3f}, mean={ratio_stats["mean"]:.3f}')
            
            # PM2.5 should typically be less than PM10, so ratio should be < 1
            high_ratio_count = (df['pm2_5_pm10_ratio'] > 1.0).sum()
            print(f'PM2.5/PM10 ratio > 1.0 count: {high_ratio_count} (suspicious)')
        
    if 'traffic_index' in df.columns:
        missing_traffic = df['traffic_index'].isnull().sum()
        print(f'Missing traffic index: {missing_traffic}')
    
    print()
    print('=== TEMPORAL FEATURE ISSUES ===')
    if 'hour' in df.columns:
        hour_range = f"{df['hour'].min()} to {df['hour'].max()}"
        print(f'Hour range: {hour_range}')
        if df['hour'].min() < 0 or df['hour'].max() > 23:
            print('❌ Invalid hour values detected!')
    
    if 'season' in df.columns:
        season_counts = df['season'].value_counts().sort_index()
        print(f'Season distribution: {dict(season_counts)}')
        if not all(s in [0, 1, 2, 3] for s in df['season'].unique()):
            print('❌ Invalid season values detected!')
    
    print()
    print('=== MISSING DATA PATTERNS ===')
    # Check for rows with all missing engineered features
    engineered_cols = ['pm2_5_pm10_ratio', 'traffic_index', 'total_pm', 'pm_weighted']
    available_engineered = [col for col in engineered_cols if col in df.columns]
    
    if available_engineered:
        all_missing = df[available_engineered].isnull().all(axis=1).sum()
        print(f'Rows with all engineered features missing: {all_missing}')
    
    # Check for early rows (likely missing lag features)
    if 'pm2_5_nowcast_lag_1h' in df.columns:
        lag_missing = df['pm2_5_nowcast_lag_1h'].isnull().sum()
        print(f'Missing lag features: {lag_missing} (expected for first few rows)')
    
    print()
    print('=== SAMPLE OF PROBLEMATIC ROWS ===')
    # Show first few rows to check for issues
    print('First 3 rows:')
    print(df.head(3)[['datetime', 'aqi_epa_calc', 'pm2_5_nowcast', 'pm10_nowcast']].to_string())
    
    return df

if __name__ == "__main__":
    df = analyze_retrieved_data()