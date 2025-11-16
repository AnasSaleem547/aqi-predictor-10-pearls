#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis (EDA) for Karachi AQI Dataset
=====================================================================

Purpose: Analyze the AQI dataset to understand:
1. Data structure and quality
2. Feature relationships and correlations
3. Temporal patterns
4. Distribution characteristics
5. Feature selection recommendations
6. Feature engineering opportunities

Author: EDA Analysis
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_examine_data(file_path):
    """Load the dataset and perform initial examination."""
    print("=" * 60)
    print("LOADING AND EXAMINING DATASET")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"✅ Dataset loaded successfully: {file_path}")
    print(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Basic info
    print("\n📋 DATASET OVERVIEW:")
    print("-" * 30)
    print(df.info())
    
    # Column names and types
    print("\n📝 COLUMN DETAILS:")
    print("-" * 30)
    for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes), 1):
        print(f"{i:2d}. {col:<20} | {dtype}")
    
    # First few rows
    print("\n👀 FIRST 5 ROWS:")
    print("-" * 30)
    print(df.head())
    
    # Last few rows
    print("\n👀 LAST 5 ROWS:")
    print("-" * 30)
    print(df.tail())
    
    return df

def analyze_data_quality(df):
    """Analyze data quality, missing values, and completeness."""
    print("\n" + "=" * 60)
    print("DATA QUALITY ANALYSIS")
    print("=" * 60)
    
    # Missing values analysis
    print("🔍 MISSING VALUES ANALYSIS:")
    print("-" * 30)
    missing_stats = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
        'Data_Type': df.dtypes
    })
    missing_stats = missing_stats.sort_values('Missing_Percentage', ascending=False)
    print(missing_stats)
    
    # Zero values analysis (potential implicit missing data)
    print("\n🔍 ZERO VALUES ANALYSIS:")
    print("-" * 30)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    zero_stats = pd.DataFrame({
        'Column': numeric_cols,
        'Zero_Count': [sum(df[col] == 0) for col in numeric_cols],
        'Zero_Percentage': [(sum(df[col] == 0) / len(df)) * 100 for col in numeric_cols]
    })
    zero_stats = zero_stats.sort_values('Zero_Percentage', ascending=False)
    print(zero_stats)
    
    # Duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\n🔍 DUPLICATE ROWS: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    # Data completeness summary
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells) * 100
    print(f"\n📊 OVERALL DATA COMPLETENESS: {completeness:.2f}%")
    
    return missing_stats, zero_stats

def analyze_descriptive_statistics(df):
    """Generate comprehensive descriptive statistics."""
    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    
    # Numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    print("📊 BASIC STATISTICS:")
    print("-" * 30)
    desc_stats = numeric_df.describe()
    print(desc_stats)
    
    # Additional statistics
    print("\n📊 ADDITIONAL STATISTICS:")
    print("-" * 30)
    additional_stats = pd.DataFrame({
        'Column': numeric_df.columns,
        'Skewness': numeric_df.skew(),
        'Kurtosis': numeric_df.kurtosis(),
        'Variance': numeric_df.var(),
        'Range': numeric_df.max() - numeric_df.min(),
        'IQR': numeric_df.quantile(0.75) - numeric_df.quantile(0.25)
    })
    print(additional_stats.round(4))
    
    return desc_stats, additional_stats

def analyze_temporal_patterns(df):
    """Analyze temporal patterns in the data."""
    print("\n" + "=" * 60)
    print("TEMPORAL PATTERNS ANALYSIS")
    print("=" * 60)
    
    # Convert datetime if it's not already
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Extract temporal features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Time range
        print(f"📅 TIME RANGE:")
        print(f"   Start: {df['datetime'].min()}")
        print(f"   End:   {df['datetime'].max()}")
        print(f"   Duration: {df['datetime'].max() - df['datetime'].min()}")
        
        # Frequency analysis
        time_diffs = df['datetime'].diff().dropna()
        most_common_freq = time_diffs.mode()[0] if not time_diffs.empty else None
        print(f"   Most common frequency: {most_common_freq}")
        
        # Hourly patterns
        print(f"\n⏰ HOURLY PATTERNS:")
        print("-" * 30)
        if 'aqi_epa_calc' in df.columns:
            hourly_aqi = df.groupby('hour')['aqi_epa_calc'].agg(['mean', 'std', 'count'])
            print("Hour | Mean AQI | Std AQI | Count")
            print("-" * 35)
            for hour in range(24):
                if hour in hourly_aqi.index:
                    mean_aqi = hourly_aqi.loc[hour, 'mean']
                    std_aqi = hourly_aqi.loc[hour, 'std']
                    count = hourly_aqi.loc[hour, 'count']
                    print(f"{hour:4d} | {mean_aqi:8.2f} | {std_aqi:7.2f} | {count:5d}")
        
        # Weekend vs Weekday
        print(f"\n📅 WEEKEND vs WEEKDAY PATTERNS:")
        print("-" * 30)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        weekend_comparison = df.groupby('is_weekend')[numeric_cols].mean()
        print("Weekend (True) vs Weekday (False) - Mean Values:")
        print(weekend_comparison.round(3))
        
    return df

def analyze_correlations(df):
    """Analyze correlations between features."""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Correlation matrix
    corr_matrix = numeric_df.corr()
    
    print("🔗 CORRELATION MATRIX:")
    print("-" * 30)
    print(corr_matrix.round(3))
    
    # Strong correlations (> 0.7 or < -0.7)
    print(f"\n🔗 STRONG CORRELATIONS (|r| > 0.7):")
    print("-" * 30)
    strong_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                strong_corrs.append({
                    'Feature_1': corr_matrix.columns[i],
                    'Feature_2': corr_matrix.columns[j],
                    'Correlation': corr_val
                })
    
    if strong_corrs:
        strong_corr_df = pd.DataFrame(strong_corrs)
        strong_corr_df = strong_corr_df.sort_values('Correlation', key=abs, ascending=False)
        print(strong_corr_df)
    else:
        print("No strong correlations found.")
    
    # Target variable correlations (if AQI is target)
    if 'aqi_epa_calc' in numeric_df.columns:
        print(f"\n🎯 CORRELATIONS WITH TARGET (aqi_epa_calc):")
        print("-" * 30)
        target_corrs = corr_matrix['aqi_epa_calc'].drop('aqi_epa_calc').sort_values(key=abs, ascending=False)
        for feature, corr in target_corrs.items():
            print(f"{feature:<20} | {corr:7.4f}")
    
    return corr_matrix, strong_corrs

def analyze_distributions(df):
    """Analyze distribution characteristics of features."""
    print("\n" + "=" * 60)
    print("DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    print("📊 DISTRIBUTION CHARACTERISTICS:")
    print("-" * 30)
    
    distribution_stats = []
    for col in numeric_df.columns:
        data = numeric_df[col].dropna()
        
        # Basic distribution stats
        stats = {
            'Feature': col,
            'Mean': data.mean(),
            'Median': data.median(),
            'Mode': data.mode().iloc[0] if not data.mode().empty else np.nan,
            'Std': data.std(),
            'Skewness': data.skew(),
            'Kurtosis': data.kurtosis(),
            'Min': data.min(),
            'Max': data.max(),
            'Q1': data.quantile(0.25),
            'Q3': data.quantile(0.75)
        }
        distribution_stats.append(stats)
    
    dist_df = pd.DataFrame(distribution_stats)
    print(dist_df.round(4))
    
    # Distribution types
    print(f"\n📊 DISTRIBUTION TYPES:")
    print("-" * 30)
    for col in numeric_df.columns:
        skew = numeric_df[col].skew()
        if abs(skew) < 0.5:
            dist_type = "Approximately Normal"
        elif skew > 0.5:
            dist_type = "Right-skewed (Positive)"
        else:
            dist_type = "Left-skewed (Negative)"
        print(f"{col:<20} | Skew: {skew:6.3f} | {dist_type}")
    
    return dist_df

def detect_outliers(df):
    """Detect outliers using IQR and Z-score methods."""
    print("\n" + "=" * 60)
    print("OUTLIER DETECTION")
    print("=" * 60)
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    print("🚨 OUTLIER ANALYSIS (IQR Method):")
    print("-" * 30)
    
    outlier_summary = []
    for col in numeric_df.columns:
        data = numeric_df[col].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(data)) * 100
        
        outlier_summary.append({
            'Feature': col,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound,
            'Outlier_Count': outlier_count,
            'Outlier_Percentage': outlier_percentage,
            'Min_Outlier': outliers.min() if outlier_count > 0 else np.nan,
            'Max_Outlier': outliers.max() if outlier_count > 0 else np.nan
        })
    
    outlier_df = pd.DataFrame(outlier_summary)
    print(outlier_df.round(3))
    
    # Z-score method
    print(f"\n🚨 EXTREME OUTLIERS (Z-score > 3):")
    print("-" * 30)
    for col in numeric_df.columns:
        data = numeric_df[col].dropna()
        z_scores = np.abs((data - data.mean()) / data.std())
        extreme_outliers = data[z_scores > 3]
        if len(extreme_outliers) > 0:
            print(f"{col}: {len(extreme_outliers)} extreme outliers ({len(extreme_outliers)/len(data)*100:.2f}%)")
            print(f"   Max Z-score: {z_scores.max():.2f}")
    
    return outlier_df

def feature_importance_analysis(df):
    """Analyze feature importance for selection."""
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if 'aqi_epa_calc' in numeric_df.columns:
        target = 'aqi_epa_calc'
        features = [col for col in numeric_df.columns if col != target]
        
        print(f"🎯 FEATURE IMPORTANCE FOR TARGET: {target}")
        print("-" * 30)
        
        # Correlation-based importance
        correlations = numeric_df[features].corrwith(numeric_df[target]).abs().sort_values(ascending=False)
        
        print("Correlation-based Ranking:")
        for i, (feature, corr) in enumerate(correlations.items(), 1):
            importance_level = "High" if corr > 0.7 else "Medium" if corr > 0.4 else "Low"
            print(f"{i:2d}. {feature:<20} | {corr:6.4f} | {importance_level}")
        
        # Variance analysis
        print(f"\n📊 FEATURE VARIANCE ANALYSIS:")
        print("-" * 30)
        variances = numeric_df[features].var().sort_values(ascending=False)
        for feature, var in variances.items():
            print(f"{feature:<20} | Variance: {var:10.4f}")
        
        return correlations, variances
    
    return None, None

def generate_feature_recommendations(df, correlations=None):
    """Generate recommendations for feature selection and engineering."""
    print("\n" + "=" * 60)
    print("FEATURE SELECTION & ENGINEERING RECOMMENDATIONS")
    print("=" * 60)
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    print("💡 FEATURE SELECTION RECOMMENDATIONS:")
    print("-" * 30)
    
    if correlations is not None:
        # High importance features
        high_importance = correlations[correlations > 0.7].index.tolist()
        medium_importance = correlations[(correlations > 0.4) & (correlations <= 0.7)].index.tolist()
        low_importance = correlations[correlations <= 0.4].index.tolist()
        
        print("🟢 HIGH IMPORTANCE (Keep):")
        for feature in high_importance:
            print(f"   ✅ {feature}")
        
        print("\n🟡 MEDIUM IMPORTANCE (Consider):")
        for feature in medium_importance:
            print(f"   ⚠️  {feature}")
        
        print("\n🔴 LOW IMPORTANCE (Consider Removing):")
        for feature in low_importance:
            print(f"   ❌ {feature}")
    
    print(f"\n💡 FEATURE ENGINEERING OPPORTUNITIES:")
    print("-" * 30)
    
    # Temporal features
    if 'datetime' in df.columns:
        print("⏰ TEMPORAL FEATURES:")
        print("   • Hour of day (rush hour indicators)")
        print("   • Day of week (weekend/weekday)")
        print("   • Month/Season indicators")
        print("   • Time-based rolling averages")
    
    # Ratio features
    if 'pm2_5_nowcast' in df.columns and 'pm10_nowcast' in df.columns:
        print("\n📊 RATIO FEATURES:")
        print("   • PM2.5/PM10 ratio (combustion indicator)")
        print("   • Fine/Coarse particle ratio")
    
    # Interaction features
    print("\n🔗 INTERACTION FEATURES:")
    print("   • Traffic pollutants combination (CO + NO2)")
    print("   • Particulate matter index (PM2.5 + PM10)")
    print("   • Photochemical potential (O3 formation indicators)")
    
    # Lag features
    print("\n⏳ LAG FEATURES:")
    print("   • Previous hour values (t-1, t-2, t-3)")
    print("   • Moving averages (3h, 6h, 12h, 24h)")
    print("   • Trend indicators (increasing/decreasing)")
    
    # Categorical features
    print("\n🏷️  CATEGORICAL FEATURES:")
    print("   • AQI category bins (Good, Moderate, Unhealthy, etc.)")
    print("   • Pollution level indicators (Low, Medium, High)")
    print("   • Rush hour flags (Morning/Evening peaks)")

def main():
    """Main EDA execution function."""
    print("🔍 COMPREHENSIVE EDA FOR KARACHI AQI DATASET")
    print("=" * 60)
    
    # Load and examine data
    df = load_and_examine_data('karachi_aqi_data_nowcast.csv')
    
    # Data quality analysis
    missing_stats, zero_stats = analyze_data_quality(df)
    
    # Descriptive statistics
    desc_stats, additional_stats = analyze_descriptive_statistics(df)
    
    # Temporal patterns
    df = analyze_temporal_patterns(df)
    
    # Correlation analysis
    corr_matrix, strong_corrs = analyze_correlations(df)
    
    # Distribution analysis
    dist_stats = analyze_distributions(df)
    
    # Outlier detection
    outlier_stats = detect_outliers(df)
    
    # Feature importance
    correlations, variances = feature_importance_analysis(df)
    
    # Recommendations
    generate_feature_recommendations(df, correlations)
    
    print("\n" + "=" * 60)
    print("✅ EDA ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return df, {
        'missing_stats': missing_stats,
        'zero_stats': zero_stats,
        'desc_stats': desc_stats,
        'additional_stats': additional_stats,
        'corr_matrix': corr_matrix,
        'strong_corrs': strong_corrs,
        'dist_stats': dist_stats,
        'outlier_stats': outlier_stats,
        'correlations': correlations,
        'variances': variances
    }

if __name__ == "__main__":
    df, results = main()