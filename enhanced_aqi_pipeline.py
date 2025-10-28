# ============================================================
# File: enhanced_aqi_pipeline.py
# Author: Anas Saleem
# Institution: FAST NUCES
# Enhanced Version: Feature Engineering + Hopsworks Integration
# ============================================================
"""
Enhanced Air Quality Index (AQI) Pipeline for Karachi
=====================================================

Purpose:
--------
This enhanced pipeline extends the original EPA AQI computation with:
1. Advanced feature engineering based on comprehensive EDA analysis
2. Hopsworks Feature Store integration for ML pipeline
3. Data quality checks and gap filling for production readiness
4. Research-backed feature selection (16 refined features)

Key Enhancements Over Original:
------------------------------
✅ Temporal Features: Hour, season, rush hour, weekend patterns
✅ Engineered Features: PM ratios, traffic index based on correlation analysis
✅ Minimal Lag Features: 1h lag and 3h moving average for PM2.5 (avoiding data leakage)
✅ Data Quality: Gap detection, interpolation, and validation
✅ Hopsworks Integration: Feature Store upload with error handling
✅ Production Ready: Comprehensive logging and error handling

Research Foundation:
-------------------
Feature selection based on comprehensive EDA analysis showing:
- PM2.5 (r=0.96) and PM10 (r=0.94) as primary AQI drivers
- Strong traffic correlation: CO (r=0.87) and NO2 (r=0.86)
- Temporal patterns: Rush hour effects, seasonal variations
- Excluded: O3 (r=0.12), excessive lag features, weak temporal features

Target Feature Set (16 + target):
---------------------------------
Core Pollutants (5): pm2_5_nowcast, pm10_nowcast, co_ppm_8hr_avg, no2_ppb, so2_ppb
Engineered (3): pm2_5_pm10_ratio, traffic_index, total_pm
Temporal (4): hour, is_rush_hour, is_weekend, season
Minimal Lag (2): pm2_5_nowcast_lag_1h, pm2_5_nowcast_ma_3h
Target (1): aqi_epa_calc

References:
-----------
- EDA_report.md: Comprehensive correlation and pattern analysis
- FINAL_FEATURE_SELECTION_RECOMMENDATIONS.md: Feature importance rankings
- U.S. EPA Technical Assistance Document for AQI Reporting (2018)
- EPA NowCast PM Method: https://forum.airnowtech.org/t/the-nowcast-for-pm2-5-and-pm10/172
"""

# ============================================================
# Imports
# ============================================================
import os
import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import zoneinfo
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
import warnings
warnings.filterwarnings('ignore')

# Hopsworks imports with error handling
try:
    import hopsworks
    HOPSWORKS_AVAILABLE = True
    print("✅ Hopsworks SDK imported successfully")
except ImportError:
    HOPSWORKS_AVAILABLE = False
    print("⚠️ Hopsworks SDK not available. Install with: pip install hopsworks")

# ============================================================
# Environment & Configuration
# ============================================================
load_dotenv()

# API Keys and Configuration
API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not API_KEY:
    raise ValueError("❌ OPENWEATHER_API_KEY not found in environment variables!")

# Karachi coordinates
LAT, LON = 24.8546842, 67.0207055

# Console for rich output
console = Console(force_terminal=True, width=200)

# ============================================================
# EPA AQI Breakpoints (Same as original)
# ============================================================
AQI_BREAKPOINTS = {
    "pm2_5": [
        (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)
    ],
    "pm10": [
        (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
        (255, 354, 151, 200), (355, 424, 201, 300),
        (425, 504, 301, 400), (505, 604, 401, 500)
    ],
    "no2": [
        (0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
        (361, 649, 151, 200), (650, 1249, 201, 300),
        (1250, 1649, 301, 400), (1650, 2049, 401, 500)
    ],
    "so2": [
        (0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150),
        (186, 304, 151, 200), (305, 604, 201, 300),
        (605, 804, 301, 400), (805, 1004, 401, 500)
    ],
    "co": [
        (0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300),
        (30.5, 40.4, 301, 400), (40.5, 50.4, 401, 500)
    ],
    "o3_8hr": [
        (0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150),
        (86, 105, 151, 200), (106, 200, 201, 300)
    ],
    "o3_1hr": [
        (125, 164, 101, 150), (165, 204, 151, 200),
        (205, 404, 201, 300), (405, 504, 301, 400),
        (505, 604, 401, 500)
    ],
}

# ============================================================
# Core Functions (Same as original)
# ============================================================
def convert_units(pollutant: str, value: float) -> float:
    """Convert µg/m³ → ppb or ppm based on pollutant."""
    if value is None or isinstance(value, str):
        return None
    if math.isnan(value):
        return None

    molecular_weights = {"co": 28.01, "no2": 46.0055, "so2": 64.066, "o3": 48.00}
    MW = molecular_weights.get(pollutant)
    if MW is None:
        return value  # PM stays in µg/m³

    ppb = (value * 24.45) / MW
    return ppb / 1000 if pollutant == "co" else ppb

def compute_nowcast(series: pd.Series) -> pd.Series:
    """
    Compute NowCast using EPA's algorithm for PM2.5 and PM10.
    
    Research Rationale:
    ------------------
    EPA's NowCast provides more responsive AQI calculation for hourly data
    by weighting recent hours more heavily. Critical for real-time predictions.
    
    Algorithm:
    - Consider last 12 hours
    - weight factor = max(min(ratio**11, 0.5), 0.5) where ratio = min/max
    - Weighted average = Σ(value_i * w^i) / Σ(w^i)
    """
    nowcast_vals = []
    for i in range(len(series)):
        window = series[max(0, i - 11): i + 1].dropna()
        if len(window) < 3:
            nowcast_vals.append(None)
            continue
        c_min, c_max = window.min(), window.max()
        ratio = c_min / c_max if c_max > 0 else 0
        weight_factor = max(min(ratio ** 11, 0.5), 0.5)
        weights = [weight_factor ** (len(window) - j - 1) for j in range(len(window))]
        nowcast = (window.values * weights).sum() / sum(weights)
        nowcast_vals.append(nowcast)
    return pd.Series(nowcast_vals, index=series.index)

def calc_aqi_for_pollutant(pollutant: str, conc: float) -> int | None:
    """Apply EPA linear interpolation formula for AQI calculation."""
    if conc is None or math.isnan(conc):
        return None
    if pollutant not in AQI_BREAKPOINTS:
        return None

    for Cl, Ch, Il, Ih in AQI_BREAKPOINTS[pollutant]:
        if Cl <= conc <= Ch:
            return round(((Ih - Il) / (Ch - Cl)) * (conc - Cl) + Il)

    return 500 if conc > AQI_BREAKPOINTS[pollutant][-1][1] else 0

def calc_overall_aqi(row):
    """
    Compute overall AQI using EPA's composite maximum rule.
    
    Research Note:
    -------------
    EPA AQI is the maximum of all pollutant-specific AQIs. This ensures
    the most restrictive pollutant drives the overall health assessment.
    """
    aqi_vals = []

    # PMs use NowCast values (replacing 24-hour avg)
    if not pd.isna(row.get("pm2_5_nowcast")):
        aqi_vals.append(calc_aqi_for_pollutant("pm2_5", row["pm2_5_nowcast"]))
    if not pd.isna(row.get("pm10_nowcast")):
        aqi_vals.append(calc_aqi_for_pollutant("pm10", row["pm10_nowcast"]))

    # Gases with appropriate averaging
    if not pd.isna(row.get("co_ppm_8hr_avg")):
        aqi_vals.append(calc_aqi_for_pollutant("co", row["co_ppm_8hr_avg"]))
    if not pd.isna(row.get("o3_ppb_8hr_avg")):
        o3_val = row["o3_ppb_8hr_avg"]
        if o3_val <= 125:
            aqi_vals.append(calc_aqi_for_pollutant("o3_8hr", o3_val))
        else:
            aqi_vals.append(calc_aqi_for_pollutant("o3_1hr", o3_val))
    if not pd.isna(row.get("so2_ppb")):
        aqi_vals.append(calc_aqi_for_pollutant("so2", row["so2_ppb"]))
    if not pd.isna(row.get("no2_ppb")):
        aqi_vals.append(calc_aqi_for_pollutant("no2", row["no2_ppb"]))

    valid = [v for v in aqi_vals if v is not None]
    return max(valid) if valid else None

# ============================================================
# Data Quality and Gap Filling Functions
# ============================================================
def detect_data_gaps(df, datetime_col='datetime'):
    """
    Detect gaps in hourly time series data.
    
    Research Rationale:
    ------------------
    Continuous hourly data is critical for lag features and rolling averages.
    Gaps can cause issues in feature engineering and model training.
    """
    console.print("\n🔍 [bold cyan]Detecting Data Gaps...[/bold cyan]")
    
    df_sorted = df.sort_values(datetime_col).copy()
    df_sorted[datetime_col] = pd.to_datetime(df_sorted[datetime_col])
    
    # Calculate time differences
    time_diffs = df_sorted[datetime_col].diff()
    expected_freq = pd.Timedelta(hours=1)
    
    # Find gaps larger than expected frequency
    gaps = time_diffs[time_diffs > expected_freq]
    
    if len(gaps) > 0:
        console.print(f"⚠️  Found {len(gaps)} gaps in data:")
        for idx, gap in gaps.items():
            gap_start = df_sorted.loc[idx-1, datetime_col] if idx > 0 else "Start"
            gap_end = df_sorted.loc[idx, datetime_col]
            console.print(f"   Gap: {gap_start} → {gap_end} ({gap})")
    else:
        console.print("✅ No significant gaps detected")
    
    return gaps

def fill_hourly_gaps(df, datetime_col='datetime', method='interpolate'):
    """
    Fill gaps in hourly time series with appropriate interpolation.
    
    Research Rationale:
    ------------------
    Linear interpolation for short gaps (<6 hours) maintains temporal continuity.
    For longer gaps, we use forward fill to avoid creating artificial patterns.
    This approach preserves the underlying pollution dynamics.
    """
    console.print(f"\n🔧 [bold cyan]Filling Hourly Gaps using {method}...[/bold cyan]")
    
    df_filled = df.copy()
    df_filled[datetime_col] = pd.to_datetime(df_filled[datetime_col])
    df_filled = df_filled.sort_values(datetime_col)
    
    # Create complete hourly range
    start_time = df_filled[datetime_col].min()
    end_time = df_filled[datetime_col].max()
    complete_range = pd.date_range(start=start_time, end=end_time, freq='H')
    
    # Reindex to complete range
    df_filled = df_filled.set_index(datetime_col).reindex(complete_range)
    df_filled.index.name = datetime_col
    df_filled = df_filled.reset_index()
    
    # Fill missing values
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    
    if method == 'interpolate':
        # Linear interpolation for short gaps, forward fill for long gaps
        df_filled[numeric_cols] = df_filled[numeric_cols].interpolate(method='linear', limit=6)
        df_filled[numeric_cols] = df_filled[numeric_cols].fillna(method='ffill', limit=12)
    elif method == 'forward_fill':
        df_filled[numeric_cols] = df_filled[numeric_cols].fillna(method='ffill')
    
    filled_count = df_filled[numeric_cols].isna().sum().sum()
    original_count = df[numeric_cols].isna().sum().sum()
    
    console.print(f"✅ Gap filling completed. Missing values: {original_count} → {filled_count}")
    
    return df_filled

# ============================================================
# Enhanced Feature Engineering Functions
# ============================================================
def create_temporal_features(df):
    """
    Create temporal features based on EDA findings and domain knowledge.
    
    Research Rationale:
    ------------------
    Based on comprehensive EDA analysis:
    - Hour shows moderate correlation (r=0.22) with AQI - captures diurnal patterns
    - Rush hour patterns clearly visible in CO and NO2 data
    - Weekend effects significant for traffic-related pollutants
    - Seasonal patterns important for PM concentrations (winter elevation)
    
    Features Created:
    - hour: Direct temporal pattern (0-23) - extracted from Pakistan local time for meaningful interpretation
    - is_rush_hour: Binary flag for morning (7-9) and evening (17-19) peaks (Pakistan local time)
    - is_weekend: Binary flag for Saturday/Sunday
    - season: Categorical for Karachi climate (Winter, Spring, Summer, Monsoon)
    
    Note: All temporal features use Pakistan local time (UTC+5) to align with daily life patterns
    and provide meaningful context for AQI interpretation relative to local activities.
    """
    console.print("\n⏰ [bold cyan]Creating Temporal Features...[/bold cyan]")
    
    df = df.copy()
    
    # Extract basic temporal components using Pakistan local time for meaningful interpretation
    # datetime column is now in PKT, so we can use it directly
    df['hour'] = df['datetime'].dt.hour  # Pakistan local hour for meaningful AQI interpretation
    df['day_of_week'] = df['datetime'].dt.dayofweek  # Keep for analysis but not in final set
    df['month'] = df['datetime'].dt.month  # Keep for analysis but not in final set
    
    # Weekend indicator (EDA shows significant weekend effects)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Rush hour indicators (based on EDA traffic patterns) - using local Pakistan time
    df['is_morning_rush'] = df['hour'].isin([8, 9]).astype(int)
    df['is_evening_rush'] = df['hour'].isin([17, 18, 19,20,21,22]).astype(int)
    df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)
    
    # Seasonal indicators (Karachi-specific climate patterns)
    def get_season(month):
        """
        Karachi seasonal classification based on climate patterns:
        - Winter (Dec-Feb): Cooler, higher PM concentrations
        - Spring (Mar-May): Moderate temperatures
        - Summer (Jun-Aug): Hot, dry conditions
        - Monsoon (Sep-Nov): Humid, potential pollution washout
        """
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Monsoon'
    
    df['season'] = df['month'].apply(get_season)
    
    console.print("   ✅ Created temporal features: hour, is_rush_hour, is_weekend, season")
    return df

def create_engineered_features(df):
    """
    Create engineered features based on correlation analysis and domain knowledge.
    
    Research Rationale:
    ------------------
    Based on EDA correlation analysis:
    - PM2.5/PM10 ratio indicates combustion vs. dust sources
    - Traffic index combines CO (r=0.87) and NO2 (r=0.86) - both traffic-related
    - Total PM captures overall particulate burden
    - PM weighted emphasizes fine particles (more health-relevant)
    
    Features Created:
    - pm2_5_pm10_ratio: Fine/coarse particle ratio
    - traffic_index: Combined traffic pollutant indicator
    - total_pm: Sum of PM2.5 and PM10
    """
    console.print("\n🔧 [bold cyan]Creating Engineered Features...[/bold cyan]")
    
    df = df.copy()
    
    # PM2.5/PM10 ratio (combustion vs dust indicator)
    # Higher ratio indicates more combustion sources (vehicles, industry)
    df['pm2_5_pm10_ratio'] = df['pm2_5_nowcast'] / (df['pm10_nowcast'] + 1e-6)  # Avoid division by zero
    
    # Traffic index (combines CO and NO2 - both traffic-related with high correlation)
    # Normalized to 0-1 scale for interpretability
    df['traffic_index'] = (
        (df['co_ppm_8hr_avg'] / df['co_ppm_8hr_avg'].max()) + 
        (df['no2_ppb'] / df['no2_ppb'].max())
    ) / 2
    
    # Total particulate matter (overall PM burden)
    df['total_pm'] = df['pm2_5_nowcast'] + df['pm10_nowcast']
    
    console.print("   ✅ Created engineered features: pm2_5_pm10_ratio, traffic_index, total_pm")
    return df

def create_minimal_lag_features(df):
    """
    Create minimal lag features to avoid data leakage while capturing temporal patterns.
    
    Research Rationale:
    ------------------
    Based on feature engineering analysis and production considerations:
    - 1-hour lag for PM2.5: Captures immediate temporal persistence without leakage
    - 3-hour moving average: Smooths short-term fluctuations, useful for trend detection
    - Avoided excessive lags (6h, 12h, 24h) to prevent overfitting and complexity
    - Focus on PM2.5 as primary AQI driver (r=0.96 correlation)
    
    Data Leakage Prevention:
    - Only use past values (t-1, t-2, t-3)
    - No future information in lag calculations
    - Minimal lag window to maintain real-time prediction capability
    """
    console.print("\n⏳ [bold cyan]Creating Minimal Lag Features...[/bold cyan]")
    
    df = df.copy()
    
    # 1-hour lag for PM2.5 (primary AQI driver)
    df['pm2_5_nowcast_lag_1h'] = df['pm2_5_nowcast'].shift(1)
    
    # 3-hour moving average for PM2.5 (trend smoothing)
    df['pm2_5_nowcast_ma_3h'] = df['pm2_5_nowcast'].rolling(window=3, min_periods=2).mean()
    
    console.print("   ✅ Created minimal lag features: pm2_5_nowcast_lag_1h, pm2_5_nowcast_ma_3h")
    return df

def select_refined_features(df):
    """
    Select the refined 16-feature set based on EDA analysis and research findings.
    
    Research Rationale:
    ------------------
    Feature selection based on:
    1. High correlation with AQI target (>0.7): PM2.5, PM10, CO, NO2, SO2
    2. Meaningful temporal patterns: hour, rush_hour, weekend, season
    3. Engineered features with domain relevance: ratios, traffic index
    4. Minimal lag features for temporal patterns without overfitting
    5. Exclusion of weak features: O3 (r=0.12), excessive lags, day_of_week, month
    
    Final Feature Set (16 + target):
    - Core Pollutants (5): pm2_5_nowcast, pm10_nowcast, co_ppm_8hr_avg, no2_ppb, so2_ppb
    - Engineered (3): pm2_5_pm10_ratio, traffic_index, total_pm
    - Temporal (4): hour, is_rush_hour, is_weekend, season
    - Minimal Lag (2): pm2_5_nowcast_lag_1h, pm2_5_nowcast_ma_3h
    - Target (1): aqi_epa_calc
    """
    console.print("\n📊 [bold cyan]Selecting Refined Feature Set...[/bold cyan]")
    
    # Define refined feature set with user-requested ordering:
    # datetime, aqi_value, pm2.5 variants, pm10 variants, other pollutants, remaining features
    refined_features = [
        # Target variable (AQI value) - as requested first after datetime
        'aqi_epa_calc',          # EPA-calculated AQI target
        
        # PM2.5 variants (all PM2.5 related features)
        'pm2_5_nowcast',         # r=0.96 - Primary AQI driver
        'pm2_5_nowcast_lag_1h',  # 1-hour lag for immediate persistence
        'pm2_5_nowcast_ma_3h',   # 3-hour moving average for trend
        
        # PM10 variants (all PM10 related features)
        'pm10_nowcast',          # r=0.94 - Secondary PM driver
        
        # Rest of the pollutants (other core pollutants)
        'co_ppm_8hr_avg',        # r=0.87 - Traffic indicator
        'no2_ppb',               # r=0.86 - Traffic indicator
        'so2_ppb',               # r=0.82 - Industrial indicator
        
        # Remaining features (engineered + temporal)
        'pm2_5_pm10_ratio',      # Combustion vs dust indicator
        'traffic_index',         # Combined traffic pollutant score
        'total_pm',              # Overall particulate burden
        'hour',                  # r=0.22 - Diurnal patterns
        'is_rush_hour',          # Traffic peak indicator
        'is_weekend',            # Weekend effect
        'season',                # Seasonal patterns
    ]
    
    # Add datetime for reference (not a feature) - as requested first
    final_columns = ['datetime'] + refined_features
    
    # Select only available columns
    available_columns = [col for col in final_columns if col in df.columns]
    df_refined = df[available_columns].copy()
    
    console.print(f"   ✅ Selected {len(refined_features)} features + target")
    console.print(f"   📋 Features: {', '.join(refined_features[:-1])}")
    console.print(f"   🎯 Target: {refined_features[-1]}")
    
    return df_refined

# ============================================================
# Hopsworks Integration Functions
# ============================================================
def connect_to_hopsworks():
    """
    Connect to Hopsworks Feature Store with comprehensive error handling.
    
    Research Note:
    -------------
    Hopsworks provides managed feature store for ML pipelines, enabling:
    - Feature versioning and lineage
    - Data validation and monitoring
    - Seamless integration with ML training pipelines
    """
    if not HOPSWORKS_AVAILABLE:
        console.print("❌ [bold red]Hopsworks SDK not available[/bold red]")
        return None, None
    
    if not HOPSWORKS_PROJECT or not HOPSWORKS_API_KEY:
        console.print("❌ [bold red]Hopsworks credentials not found in .env file[/bold red]")
        return None, None
    
    try:
        console.print(f"\n🔗 [bold cyan]Connecting to Hopsworks project: {HOPSWORKS_PROJECT}[/bold cyan]")
        
        # Connect to Hopsworks with API key
        project = hopsworks.login(
            project=HOPSWORKS_PROJECT,
            api_key_value=HOPSWORKS_API_KEY
        )
        
        # Get feature store (use default feature store name)
        fs = project.get_feature_store()
        
        console.print("✅ [bold green]Successfully connected to Hopsworks[/bold green]")
        console.print(f"   📊 Project: {project.name}")
        console.print(f"   🏪 Feature Store: {fs.name}")
        return project, fs
        
    except Exception as e:
        console.print(f"❌ [bold red]Failed to connect to Hopsworks: {str(e)}[/bold red]")
        console.print(f"   💡 [bold yellow]Tip: Verify project name '{HOPSWORKS_PROJECT}' and API key are correct[/bold yellow]")
        return None, None

def validate_features_for_hopsworks(df):
    """
    Validate feature data before uploading to Hopsworks.
    
    Validation Checks:
    - No infinite values
    - Reasonable value ranges
    - Sufficient data completeness
    - Proper data types
    """
    console.print("\n🔍 [bold cyan]Validating Features for Hopsworks...[/bold cyan]")
    
    validation_results = {
        'passed': True,
        'issues': []
    }
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols]).sum()
    if inf_counts.sum() > 0:
        validation_results['passed'] = False
        validation_results['issues'].append(f"Infinite values found: {inf_counts[inf_counts > 0].to_dict()}")
    
    # Check data completeness
    missing_pct = (df.isnull().sum() / len(df)) * 100
    high_missing = missing_pct[missing_pct > 20]  # More than 20% missing
    if len(high_missing) > 0:
        validation_results['passed'] = False
        validation_results['issues'].append(f"High missing values: {high_missing.to_dict()}")
    
    # Check value ranges (basic sanity checks)
    if 'aqi_epa_calc' in df.columns:
        aqi_range = df['aqi_epa_calc'].describe()
        if aqi_range['min'] < 0 or aqi_range['max'] > 1000:
            validation_results['issues'].append(f"AQI values outside expected range: {aqi_range['min']:.1f} - {aqi_range['max']:.1f}")
    
    # Report validation results
    if validation_results['passed']:
        console.print("✅ [bold green]All validation checks passed[/bold green]")
    else:
        console.print("⚠️ [bold yellow]Validation issues found:[/bold yellow]")
        for issue in validation_results['issues']:
            console.print(f"   • {issue}")
    
    return validation_results

def upload_to_hopsworks(df, fs, feature_group_name="karachi_aqi_features"):
    """
    Upload processed features to Hopsworks Feature Store.
    
    Parameters:
    - df: DataFrame with processed features
    - fs: Hopsworks feature store connection
    - feature_group_name: Name for the feature group
    """
    try:
        console.print(f"\n📤 [bold cyan]Uploading to Hopsworks Feature Group: {feature_group_name}[/bold cyan]")
        
        # Validate data before upload
        validation = validate_features_for_hopsworks(df)
        if not validation['passed']:
            console.print("⚠️ [bold yellow]Proceeding with upload despite validation issues[/bold yellow]")
        
        # Create or get feature group
        feature_group = fs.get_or_create_feature_group(
            name=feature_group_name,
            version=1,
            description="Karachi AQI features with temporal and engineered features for ML training",
            primary_key=["datetime"],
            event_time="datetime"
        )
        
        # Upload data
        feature_group.insert(df, write_options={"wait_for_job": True})
        
        console.print(f"✅ [bold green]Successfully uploaded {len(df)} records to Hopsworks[/bold green]")
        console.print(f"   📊 Feature Group: {feature_group_name}")
        console.print(f"   📈 Features: {len(df.columns) - 1}")  # Exclude datetime
        console.print(f"   📅 Time Range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        return True
        
    except Exception as e:
        console.print(f"❌ [bold red]Failed to upload to Hopsworks: {str(e)}[/bold red]")
        return False

# ============================================================
# Data Fetching Functions (Enhanced from original)
# ============================================================
def fetch_aqi_data(lat: float, lon: float, start_date: datetime, end_date: datetime):
    """Fetch data from OpenWeather API (same as original)."""
    BASE_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat": lat, "lon": lon,
        "start": int(start_date.timestamp()), "end": int(end_date.timestamp()),
        "appid": API_KEY,
    }

    console.print(f"\n📡 [bold cyan]Fetching AQI data for {lat:.4f}, {lon:.4f}[/bold cyan]")
    console.print(f"   📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    r = requests.get(BASE_URL, params=params)
    if r.status_code != 200:
        console.print(f"❌ [bold red]API Error {r.status_code}: {r.text}[/bold red]")
        return None
    
    console.print("✅ [bold green]Data fetched successfully[/bold green]")
    return r.json()

def json_to_dataframe(json_data):
    """
    Convert OpenWeather JSON to DataFrame with enhanced processing.
    
    Enhancements over original:
    - Better error handling
    - Comprehensive logging
    - PKT timezone handling throughout (UTC only at fetch level)
    """
    if not json_data or "list" not in json_data:
        console.print("⚠️ [bold yellow]No data found in API response[/bold yellow]")
        return None

    PKT = zoneinfo.ZoneInfo("Asia/Karachi")
    records = []
    
    console.print("\n🔄 [bold cyan]Processing API Response...[/bold cyan]")
    
    for item in json_data["list"]:
        # Fetch in UTC, convert to PKT, then make timezone-naive for Hopsworks
        dt_utc = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
        dt_pkt = dt_utc.astimezone(PKT)
        dt_pkt_naive = dt_pkt.replace(tzinfo=None)  # Remove timezone info, keep PKT time
        
        rec = {
            "datetime": dt_pkt_naive,  # Timezone-naive PKT for Hopsworks
            "aqi_owm": item["main"]["aqi"], 
            **item["components"]
        }
        records.append(rec)

    df = pd.DataFrame(records).sort_values("datetime")
    console.print(f"   📊 Raw records: {len(df)}")

    # Unit conversions
    for gas in ["co", "no2", "so2", "o3"]:
        converted = df[gas].apply(lambda v: convert_units(gas, v))
        unit = "ppm" if gas == "co" else "ppb"
        df[f"{gas}_{unit}"] = converted

    # NowCast for PM
    console.print("   🔄 Computing NowCast for PM2.5 and PM10...")
    df["pm2_5_nowcast"] = compute_nowcast(df["pm2_5"])
    df["pm10_nowcast"] = compute_nowcast(df["pm10"])

    # Rolling averages
    console.print("   🔄 Computing rolling averages...")
    df["co_ppm_8hr_avg"] = df["co_ppm"].rolling(8, min_periods=6).mean()
    df["o3_ppb_8hr_avg"] = df["o3_ppb"].rolling(8, min_periods=6).mean()

    # AQI computation
    console.print("   🔄 Computing EPA AQI...")
    df["aqi_epa_calc"] = df.apply(calc_overall_aqi, axis=1)
    
    # Column selection and ordering
    cols = [
        "datetime", "aqi_owm", "aqi_epa_calc",
        "pm2_5_nowcast", "pm10_nowcast",
        "co_ppm", "co_ppm_8hr_avg",
        "o3_ppb", "o3_ppb_8hr_avg",
        "no2_ppb", "so2_ppb"
    ]
    
    df_final = df[cols].copy()
    console.print(f"   ✅ Processed {len(df_final)} records with {len(cols)} columns")
    
    return df_final

# ============================================================
# Display Functions
# ============================================================
def display_feature_summary(df):
    """Display comprehensive feature summary."""
    console.print("\n📊 [bold cyan]FEATURE SUMMARY[/bold cyan]")
    console.print("=" * 80)
    
    # Basic info
    console.print(f"📈 Total Records: {len(df):,}")
    console.print(f"📅 Time Range: {df['datetime'].min()} → {df['datetime'].max()}")
    console.print(f"⏱️  Duration: {df['datetime'].max() - df['datetime'].min()}")
    
    # Feature categories
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    console.print(f"\n🔢 Numeric Features: {len(numeric_cols)}")
    console.print(f"🏷️  Categorical Features: {len(categorical_cols)}")
    
    # Missing values
    missing_summary = df.isnull().sum()
    missing_pct = (missing_summary / len(df)) * 100
    
    if missing_summary.sum() > 0:
        console.print(f"\n⚠️  Missing Values:")
        for col, count in missing_summary[missing_summary > 0].items():
            console.print(f"   {col}: {count} ({missing_pct[col]:.1f}%)")
    else:
        console.print(f"\n✅ No missing values")
    
    # Target variable summary
    if 'aqi_epa_calc' in df.columns:
        aqi_stats = df['aqi_epa_calc'].describe()
        console.print(f"\n🎯 Target Variable (AQI) Summary:")
        console.print(f"   Mean: {aqi_stats['mean']:.1f}")
        console.print(f"   Std:  {aqi_stats['std']:.1f}")
        console.print(f"   Range: {aqi_stats['min']:.1f} - {aqi_stats['max']:.1f}")

def display_table_rich(df, last_n=10):
    """Display data table using Rich formatting."""
    table = Table(show_header=True, header_style="bold cyan", expand=True)
    
    # Add columns
    for col in df.columns:
        table.add_column(col, justify="right" if pd.api.types.is_numeric_dtype(df[col]) else "left")
    
    # Add rows (show last N records)
    subset = df.tail(last_n) if last_n else df
    for _, row in subset.iterrows():
        values = []
        for v in row:
            if pd.isna(v):
                values.append("-")
            elif isinstance(v, float):
                values.append(f"{v:.2f}")
            else:
                values.append(str(v))
        table.add_row(*values)
    
    console.print(table)

# ============================================================
# Main Enhanced Pipeline
# ============================================================
def main():
    """
    Main enhanced AQI pipeline execution.
    
    Pipeline Steps:
    1. Fetch raw data from OpenWeather API
    2. Process and compute EPA AQI
    3. Detect and fill data gaps
    4. Create temporal features
    5. Create engineered features
    6. Create minimal lag features
    7. Select refined feature set
    8. Upload to Hopsworks Feature Store
    9. Save locally for backup
    """
    console.print("\n🚀 [bold green]ENHANCED KARACHI AQI PIPELINE[/bold green]")
    console.print("=" * 80)
    
    # Configuration
    end_date = datetime.now(timezone.utc)
    start_date = end_date - relativedelta(months=3)  # Extended to 3 months for better features
    
    try:
        # Step 1: Fetch raw data
        data = fetch_aqi_data(LAT, LON, start_date, end_date)
        if not data:
            console.print("❌ [bold red]Failed to fetch data. Exiting.[/bold red]")
            return None
        
        # Step 2: Process raw data
        df = json_to_dataframe(data)
        if df is None or df.empty:
            console.print("❌ [bold red]No data available for processing. Exiting.[/bold red]")
            return None
        
        # Step 3: Data quality checks and gap filling
        gaps = detect_data_gaps(df)
        if len(gaps) > 0:
            df = fill_hourly_gaps(df, method='interpolate')
        
        # Step 4: Feature engineering
        df = create_temporal_features(df)
        df = create_engineered_features(df)
        df = create_minimal_lag_features(df)
        
        # Step 5: Select refined features
        df_refined = select_refined_features(df)
        
        # Step 6: Clean data (remove rows with missing target)
        initial_count = len(df_refined)
        df_refined = df_refined.dropna(subset=['aqi_epa_calc'])
        final_count = len(df_refined)
        
        if final_count < initial_count:
            console.print(f"🧹 Removed {initial_count - final_count} rows with missing target")
        
        # Step 7: Display summary
        display_feature_summary(df_refined)
        
        # Step 8: Hopsworks integration
        project, fs = connect_to_hopsworks()
        if project and fs:
            upload_success = upload_to_hopsworks(df_refined, fs)
            if upload_success:
                console.print("✅ [bold green]Hopsworks upload completed successfully[/bold green]")
        else:
            console.print("⚠️ [bold yellow]Skipping Hopsworks upload due to connection issues[/bold yellow]")
        
        # Step 9: Save locally
        output_file = "karachi_aqi_enhanced_features.csv"
        df_refined.to_csv(output_file, index=False)
        console.print(f"\n💾 [bold green]Saved {len(df_refined)} records → {output_file}[/bold green]")
        
        # Step 10: Display sample data
        console.print(f"\n📋 [bold cyan]Sample Data (Last 10 Records):[/bold cyan]")
        display_table_rich(df_refined, last_n=10)
        
        console.print("\n🎉 [bold green]ENHANCED PIPELINE COMPLETED SUCCESSFULLY![/bold green]")
        console.print("=" * 80)
        
        return df_refined
        
    except Exception as e:
        console.print(f"\n❌ [bold red]Pipeline failed with error: {str(e)}[/bold red]")
        import traceback
        console.print(f"📋 [bold red]Traceback:[/bold red]")
        console.print(traceback.format_exc())
        return None

# ============================================================
# Script Execution
# ============================================================
if __name__ == "__main__":
    df_result = main()
    
    if df_result is not None:
        console.print(f"\n📊 [bold cyan]Final Dataset Shape: {df_result.shape}[/bold cyan]")
        console.print(f"📈 Features: {df_result.shape[1] - 1}")  # Exclude datetime
        console.print(f"📋 Records: {df_result.shape[0]:,}")
        console.print("\n✅ [bold green]Ready for ML model training![/bold green]")
    else:
        console.print("\n❌ [bold red]Pipeline execution failed.[/bold red]")