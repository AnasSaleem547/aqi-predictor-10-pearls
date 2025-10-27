#!/usr/bin/env python3
"""
Feature Engineering Script for Karachi AQI Dataset
==================================================

This script implements the feature engineering recommendations from the EDA analysis.
It creates new features for improved AQI prediction model performance.

Based on EDA findings:
- PM2.5 and PM10 are strongest predictors (r > 0.94)
- Traffic pollutants (CO, NO2) show strong correlations (r > 0.85)
- Ozone features have weak correlation (r ≈ 0.12) - excluded
- Temporal patterns provide additional predictive power

Author: Feature Engineering Analysis
Date: 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path='karachi_aqi_data_nowcast.csv'):
    """Load the AQI dataset and prepare for feature engineering."""
    print("🔄 Loading AQI dataset...")
    df = pd.read_csv(file_path)
    
    # Convert datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    print(f"✅ Loaded {len(df)} records with {len(df.columns)} columns")
    return df

def create_core_features(df):
    """Create core engineered features based on EDA recommendations."""
    print("\n🔧 Creating core engineered features...")
    
    # 1. PM Ratio Features (High Priority)
    print("   📊 PM Ratio Features...")
    df['pm2_5_pm10_ratio'] = df['pm2_5_nowcast'] / df['pm10_nowcast']
    df['fine_coarse_ratio'] = df['pm2_5_nowcast'] / (df['pm10_nowcast'] - df['pm2_5_nowcast'])
    
    # Handle division by zero
    df['fine_coarse_ratio'] = df['fine_coarse_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # 2. Traffic Pollution Index (High Priority)
    print("   🚗 Traffic Pollution Features...")
    df['traffic_index'] = (df['co_ppm_8hr_avg'] + df['no2_ppb']) / 2
    
    # Normalized traffic score (0-1 scale)
    df['traffic_score'] = (
        (df['co_ppm_8hr_avg'] / df['co_ppm_8hr_avg'].max()) + 
        (df['no2_ppb'] / df['no2_ppb'].max())
    ) / 2
    
    # 3. Combined Particulate Matter Index
    print("   🌫️  Particulate Matter Features...")
    df['total_pm'] = df['pm2_5_nowcast'] + df['pm10_nowcast']
    df['pm_weighted'] = (df['pm2_5_nowcast'] * 2 + df['pm10_nowcast']) / 3  # Weight PM2.5 more
    
    # 4. Pollution Severity Categories
    print("   📈 Pollution Severity Features...")
    df['pm2_5_category'] = pd.cut(df['pm2_5_nowcast'], 
                                  bins=[0, 12, 35.4, 55.4, 150.4, np.inf],
                                  labels=['Good', 'Moderate', 'USG', 'Unhealthy', 'Very_Unhealthy'])
    
    df['aqi_category'] = pd.cut(df['aqi_epa_calc'],
                                bins=[0, 50, 100, 150, 200, 300, np.inf],
                                labels=['Good', 'Moderate', 'USG', 'Unhealthy', 'Very_Unhealthy', 'Hazardous'])
    
    print(f"   ✅ Created {4} core feature groups")
    return df

def create_temporal_features(df):
    """Create temporal features based on datetime patterns."""
    print("\n⏰ Creating temporal features...")
    
    # Extract basic temporal components
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear
    
    # Weekend indicator
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Rush hour indicators (based on EDA findings)
    df['is_morning_rush'] = df['hour'].isin([7, 8, 9]).astype(int)
    df['is_evening_rush'] = df['hour'].isin([17, 18, 19]).astype(int)
    df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)
    
    # Time of day categories
    def categorize_time(hour):
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        elif 18 <= hour < 22:
            return 'Evening'
        else:
            return 'Night'
    
    df['time_of_day'] = df['hour'].apply(categorize_time)
    
    # Seasonal indicators
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Monsoon'
    
    df['season'] = df['month'].apply(get_season)
    
    print(f"   ✅ Created temporal features")
    return df

def create_lag_features(df, target_cols=['pm2_5_nowcast', 'aqi_epa_calc'], lags=[1, 3, 6, 12, 24]):
    """Create lag features for time series patterns."""
    print("\n⏳ Creating lag features...")
    
    for col in target_cols:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
    
    print(f"   ✅ Created lag features for {len(target_cols)} variables")
    return df

def create_rolling_features(df, target_cols=['pm2_5_nowcast', 'aqi_epa_calc'], windows=[3, 6, 12, 24]):
    """Create rolling average and statistical features."""
    print("\n📊 Creating rolling statistical features...")
    
    for col in target_cols:
        if col in df.columns:
            for window in windows:
                # Rolling averages
                df[f'{col}_ma_{window}h'] = df[col].rolling(window, min_periods=1).mean()
                
                # Rolling standard deviation (volatility)
                df[f'{col}_std_{window}h'] = df[col].rolling(window, min_periods=1).std()
                
                # Rolling min/max
                df[f'{col}_min_{window}h'] = df[col].rolling(window, min_periods=1).min()
                df[f'{col}_max_{window}h'] = df[col].rolling(window, min_periods=1).max()
    
    print(f"   ✅ Created rolling features for {len(target_cols)} variables")
    return df

def create_trend_features(df, target_cols=['pm2_5_nowcast', 'aqi_epa_calc']):
    """Create trend and change features."""
    print("\n📈 Creating trend features...")
    
    for col in target_cols:
        if col in df.columns:
            # Rate of change
            df[f'{col}_change_1h'] = df[col].diff()
            df[f'{col}_change_3h'] = df[col].diff(3)
            
            # Percentage change
            df[f'{col}_pct_change_1h'] = df[col].pct_change()
            
            # Trend indicators
            df[f'{col}_increasing'] = (df[f'{col}_change_1h'] > 0).astype(int)
            df[f'{col}_decreasing'] = (df[f'{col}_change_1h'] < 0).astype(int)
            
            # Spike detection (above 90th percentile of changes)
            threshold = df[f'{col}_change_1h'].quantile(0.9)
            df[f'{col}_spike'] = (df[f'{col}_change_1h'] > threshold).astype(int)
    
    print(f"   ✅ Created trend features for {len(target_cols)} variables")
    return df

def create_interaction_features(df):
    """Create interaction features between key variables."""
    print("\n🔗 Creating interaction features...")
    
    # PM interactions
    df['pm2_5_x_traffic'] = df['pm2_5_nowcast'] * df['traffic_index']
    df['pm10_x_traffic'] = df['pm10_nowcast'] * df['traffic_index']
    
    # Rush hour interactions
    df['pm2_5_x_rush'] = df['pm2_5_nowcast'] * df['is_rush_hour']
    df['traffic_x_rush'] = df['traffic_index'] * df['is_rush_hour']
    
    # Seasonal interactions
    df['pm2_5_x_winter'] = df['pm2_5_nowcast'] * (df['season'] == 'Winter').astype(int)
    df['traffic_x_winter'] = df['traffic_index'] * (df['season'] == 'Winter').astype(int)
    
    print(f"   ✅ Created interaction features")
    return df

def select_features_by_importance(df, target='aqi_epa_calc'):
    """Select features based on EDA correlation analysis."""
    print(f"\n🎯 Selecting features based on correlation with {target}...")
    
    # Core high-importance features (correlation > 0.7)
    core_features = [
        'pm2_5_nowcast',      # r = 0.9611
        'pm10_nowcast',       # r = 0.9449  
        'co_ppm_8hr_avg',     # r = 0.8666
        'co_ppm',             # r = 0.8649
        'no2_ppb',            # r = 0.8563
        'so2_ppb',            # r = 0.8160
        'aqi_owm'             # r = 0.7675
    ]
    
    # Engineered features (high priority)
    engineered_features = [
        'pm2_5_pm10_ratio',
        'traffic_index',
        'total_pm',
        'pm_weighted'
    ]
    
    # Temporal features (medium priority)
    temporal_features = [
        'hour',
        'is_rush_hour',
        'is_weekend',
        'season'
    ]
    
    # Lag features (medium priority)
    lag_features = [
        'pm2_5_nowcast_lag_1h',
        'aqi_epa_calc_lag_1h',
        'pm2_5_nowcast_ma_3h'
    ]
    
    # Create feature sets
    feature_sets = {
        'minimal': core_features[:3] + [target],  # Top 3 + target
        'core': core_features + [target],         # All high-correlation + target
        'enhanced': core_features + engineered_features + [target],  # Core + engineered
        'full': core_features + engineered_features + temporal_features + lag_features + [target]  # All features
    }
    
    # Filter available features
    available_features = {}
    for set_name, features in feature_sets.items():
        available = [f for f in features if f in df.columns]
        available_features[set_name] = available
        print(f"   📋 {set_name.upper()} set: {len(available)} features")
    
    return available_features

def create_model_ready_datasets(df, feature_sets, target='aqi_epa_calc'):
    """Create clean datasets ready for modeling."""
    print(f"\n🧹 Creating model-ready datasets...")
    
    datasets = {}
    
    for set_name, features in feature_sets.items():
        # Select features
        dataset = df[features].copy()
        
        # Handle missing values
        initial_rows = len(dataset)
        dataset = dataset.dropna()
        final_rows = len(dataset)
        
        # Encode categorical variables
        categorical_cols = dataset.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col != target:  # Don't encode target if it's categorical
                dataset = pd.get_dummies(dataset, columns=[col], prefix=col, drop_first=True)
        
        datasets[set_name] = dataset
        print(f"   📊 {set_name.upper()}: {final_rows} rows, {len(dataset.columns)} features ({initial_rows - final_rows} rows dropped)")
    
    return datasets

def main():
    """Main feature engineering pipeline."""
    print("🚀 FEATURE ENGINEERING PIPELINE FOR KARACHI AQI DATASET")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Create features
    df = create_core_features(df)
    df = create_temporal_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_trend_features(df)
    df = create_interaction_features(df)
    
    # Select feature sets
    feature_sets = select_features_by_importance(df)
    
    # Create model-ready datasets
    datasets = create_model_ready_datasets(df, feature_sets)
    
    # Save datasets
    print(f"\n💾 Saving feature-engineered datasets...")
    for set_name, dataset in datasets.items():
        filename = f'karachi_aqi_{set_name}_features.csv'
        dataset.to_csv(filename, index=False)
        print(f"   ✅ Saved: {filename} ({len(dataset)} rows × {len(dataset.columns)} cols)")
    
    # Save full feature-engineered dataset
    df.to_csv('karachi_aqi_all_features.csv', index=False)
    print(f"   ✅ Saved: karachi_aqi_all_features.csv ({len(df)} rows × {len(df.columns)} cols)")
    
    # Summary statistics
    print(f"\n📊 FEATURE ENGINEERING SUMMARY:")
    print(f"   Original features: 11")
    print(f"   Total features created: {len(df.columns)}")
    print(f"   Feature sets available: {len(datasets)}")
    
    print(f"\n🎯 RECOMMENDED USAGE:")
    print(f"   • Start with 'core' dataset for baseline models")
    print(f"   • Use 'enhanced' dataset for improved performance") 
    print(f"   • Try 'full' dataset for advanced models")
    print(f"   • Use 'minimal' dataset for quick prototyping")
    
    print(f"\n✅ Feature engineering completed successfully!")
    print("=" * 60)
    
    return df, datasets, feature_sets

if __name__ == "__main__":
    df, datasets, feature_sets = main()