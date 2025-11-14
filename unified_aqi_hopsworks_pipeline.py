# ============================================================
# File: unified_aqi_hopsworks_pipeline.py
# Author: Anas Saleem
# Institution: FAST NUCES
# ============================================================
"""
Purpose:
--------
Unified AQI pipeline that integrates data fetching, EPA AQI calculation, 
feature engineering, and Hopsworks feature store management for Karachi air quality data.

Features:
---------
1. Fetch air pollution data from OpenWeather API
2. Calculate EPA-style AQI with NowCast for PM pollutants
3. Advanced feature engineering (14 features + 1 target)
4. Hopsworks integration for feature store management
5. Dual pipeline modes: backfill (12 months) and hourly updates
6. Data quality validation and imputation
7. Pakistan timezone preservation

Pipeline Modes:
---------------
- backfill_pipeline(): Initialize with 12 months of historical data
- hourly_pipeline(): Incremental updates with lag feature support

Important Flags:
----------------
--force: Use this flag with backfill_pipeline() to force re-processing of data 
         even if it already exists in Hopsworks. This is essential when making
         schema changes or when you want to completely recreate the feature group.
"""

# ============================================================
# Imports
# ============================================================
import os
import math
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import zoneinfo
from typing import Optional, Tuple, Dict, Any, List

from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
import hopsworks

# ============================================================
# Environment & Configuration
# ============================================================
load_dotenv()

# API Configuration
API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not API_KEY:
    raise ValueError("❌ OPENWEATHER_API_KEY not found in environment variables!")

# Hopsworks Configuration
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT")

# Allow script to run without Hopsworks credentials for testing
HOPSWORKS_AVAILABLE = bool(HOPSWORKS_API_KEY and HOPSWORKS_PROJECT_NAME)
if not HOPSWORKS_AVAILABLE:
    print("⚠️ Hopsworks credentials not found. Running in test mode (no Hopsworks integration).")

# Location Configuration
LAT, LON = 24.8546842, 67.0207055  # Karachi coordinates
PKT = zoneinfo.ZoneInfo("Asia/Karachi")

# Feature Group Configuration 
FEATURE_GROUP_NAME = "karachifeatures10" 

# Data Quality Thresholds
MISSING_DATA_THRESHOLD = 0.30  # 30% missing data tolerance
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 5  # seconds

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
# Production Feature Schema for Hopsworks
# ============================================================
PRODUCTION_FEATURES_SCHEMA = {
    # Primary key and target
    "datetime": "timestamp",
    "datetime_id": "bigint",  # Unix timestamp as primary key for online compatibility
    "aqi_epa_calc": "int",
    
    # Core pollutants (5 features - excluding o3 due to low correlation)
    "pm2_5_nowcast": "float",
    "pm10_nowcast": "float", 
    "co_ppm_8hr_avg": "float",
    "no2_ppb": "float",
    "so2_ppb": "float",
    
    # Pollutant lag features (4 features - properly created to prevent data leakage)
    "pm2_5_nowcast_lag_1h": "float",
    "pm10_nowcast_lag_1h": "float",
    "co_ppm_8hr_avg_lag_1h": "float",
    "no2_ppb_lag_1h": "float",
    
    # Engineered features (2 features)
    "pm2_5_pm10_ratio": "float",
    "traffic_index": "float",
    
    # Temporal features (4 features)
    "hour": "int",
    "is_rush_hour": "bigint",
    "is_weekend": "bigint", 
    "season": "bigint",
    
    # Cyclical temporal encodings (4 features)
    "hour_sin": "float",
    "hour_cos": "float",
    "day_of_week_sin": "float",
    "day_of_week_cos": "float",
    
    # AQI lag features REMOVED to prevent data leakage
    # AQI is calculated from pollutants, so using AQI lag features would require
    # pollutant measurements to be available, which may not be the case in real-time
    
    # Rolling mean features (3 features - 6-hour window)
    "pm2_5_nowcast_ma_6h": "float",
    "pm10_nowcast_ma_6h": "float",
    "no2_ppb_ma_6h": "float"
}

# ============================================================
# Core Functions (From Original Script)
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
    Compute NowCast using EPA's algorithm.
    Works on hourly µg/m³ data for PM2.5 and PM10.
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
    """Apply EPA linear interpolation formula."""
    if conc is None or math.isnan(conc):
        return None
    if pollutant not in AQI_BREAKPOINTS:
        return None

    for Cl, Ch, Il, Ih in AQI_BREAKPOINTS[pollutant]:
        if Cl <= conc <= Ch:
            return round(((Ih - Il) / (Ch - Cl)) * (conc - Cl) + Il)

    return 500 if conc > AQI_BREAKPOINTS[pollutant][-1][1] else 0

def calc_overall_aqi(row):
    """Compute pollutant-specific AQIs and return the maximum (EPA rule)."""
    aqi_vals = []

    # PMs use NowCast values
    if not pd.isna(row.get("pm2_5_nowcast")):
        aqi_vals.append(calc_aqi_for_pollutant("pm2_5", row["pm2_5_nowcast"]))
    if not pd.isna(row.get("pm10_nowcast")):
        aqi_vals.append(calc_aqi_for_pollutant("pm10", row["pm10_nowcast"]))

    # Gases (excluding o3 due to low correlation)
    if not pd.isna(row.get("co_ppm_8hr_avg")):
        aqi_vals.append(calc_aqi_for_pollutant("co", row["co_ppm_8hr_avg"]))
    if not pd.isna(row.get("so2_ppb")):
        aqi_vals.append(calc_aqi_for_pollutant("so2", row["so2_ppb"]))
    if not pd.isna(row.get("no2_ppb")):
        aqi_vals.append(calc_aqi_for_pollutant("no2", row["no2_ppb"]))

    valid = [v for v in aqi_vals if v is not None]
    return max(valid) if valid else None

# ============================================================
# API Functions with Retry Logic
# ============================================================
def fetch_aqi_data_with_retry(lat: float, lon: float, start_date: datetime, end_date: datetime) -> Optional[Dict]:
    """Fetch AQI data with retry logic for API failures."""
    BASE_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat": lat, "lon": lon,
        "start": int(start_date.timestamp()), "end": int(end_date.timestamp()),
        "appid": API_KEY,
    }

    print(f"📡 Fetching AQI data for {lat:.4f}, {lon:.4f} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
    
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            r = requests.get(BASE_URL, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if data and "list" in data and len(data["list"]) > 0:
                    print(f"✅ Data fetched successfully ({len(data['list'])} records)")
                    return data
            else:
                print(f"❌ API Error {r.status_code} (attempt {attempt + 1}/{API_RETRY_ATTEMPTS})")
        except Exception as e:
            print(f"❌ Request failed: {str(e)} (attempt {attempt + 1}/{API_RETRY_ATTEMPTS})")
        
        if attempt < API_RETRY_ATTEMPTS - 1:
            import time
            time.sleep(API_RETRY_DELAY)
    
    print("❌ All API retry attempts failed")
    return None

# ============================================================
# Data Processing Functions
# ============================================================
def json_to_dataframe(json_data) -> Optional[pd.DataFrame]:
    """Convert OpenWeather JSON → DataFrame with timezone preservation."""
    if not json_data or "list" not in json_data:
        print("⚠️ No data found in API response.")
        return None

    records = []
    for item in json_data["list"]:
        # Convert to Pakistan time but preserve timezone info
        dt = datetime.fromtimestamp(item["dt"], tz=timezone.utc).astimezone(PKT)
        rec = {"datetime": dt, "aqi_owm": item["main"]["aqi"], **item["components"]}
        records.append(rec)

    df = pd.DataFrame(records).sort_values("datetime")

    # Convert gas units
    for gas in ["co", "no2", "so2", "o3"]:
        converted = df[gas].apply(lambda v: convert_units(gas, v))
        unit = "ppm" if gas == "co" else "ppb"
        df[f"{gas}_{unit}"] = converted

    # NowCast for PM
    df["pm2_5_nowcast"] = compute_nowcast(df["pm2_5"])
    df["pm10_nowcast"] = compute_nowcast(df["pm10"])

    # Rolling averages for gases
    df["co_ppm_8hr_avg"] = df["co_ppm"].rolling(8, min_periods=8).mean()
    df["o3_ppb_8hr_avg"] = df["o3_ppb"].rolling(8, min_periods=8).mean()

    # AQI computation
    df["aqi_epa_calc"] = df.apply(calc_overall_aqi, axis=1)
    
    # Convert to timezone-naive PKT for consistency with gap detection
    df["datetime"] = df["datetime"].dt.tz_localize(None)

    return df

# ============================================================
# Feature Engineering Functions
# ============================================================

def create_cyclical_features(df: pd.DataFrame, column: str, period: int) -> pd.DataFrame:
    """
    Create cyclical sine and cosine features for temporal data.
    
    Args:
        df: Input DataFrame
        column: Column name to create cyclical features from (e.g., 'hour', 'day_of_week')
        period: The period of the cycle (e.g., 24 for hours, 7 for days of week)
    
    Returns:
        DataFrame with added sin and cos columns
    """
    if column not in df.columns:
        print(f"⚠️  Column '{column}' not found in DataFrame, skipping cyclical encoding")
        return df
    
    # Convert to radians and create cyclical features
    radians = 2 * np.pi * df[column] / period
    df[f'{column}_sin'] = np.sin(radians)
    df[f'{column}_cos'] = np.cos(radians)
    
    return df

def validate_input_data(df: pd.DataFrame) -> tuple[bool, list[str]]:
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

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features for a DataFrame."""
    df = df.copy()
    
    # Temporal features (safe as they only depend on datetime)
    df['hour'] = df['datetime'].dt.hour.astype(np.int32)
    df['day_of_week'] = df['datetime'].dt.dayofweek.astype(np.int32)
    df['is_rush_hour'] = (
        (df['hour'].isin([7, 8, 9])) | 
        (df['hour'].isin([17, 18, 19, 20, 21, 22]))
    ).astype(np.int64)
    df['is_weekend'] = (df['datetime'].dt.weekday >= 5).astype(np.int64)
    
    month = df['datetime'].dt.month
    df['season'] = np.select([
        month.isin([12, 1, 2]),  # Winter
        month.isin([3, 4, 5]),   # Spring  
        month.isin([6, 7, 8]),   # Summer
        month.isin([9, 10, 11])  # Monsoon
    ], [0, 1, 2, 3], default=0).astype(np.int64)
    
    # Cyclical temporal encodings (safe)
    df = create_cyclical_features(df, 'hour', 24)
    df = create_cyclical_features(df, 'day_of_week', 7)
    
    return df

def safe_process_gap_data(df: pd.DataFrame, gap_start: datetime, gap_end: datetime) -> pd.DataFrame:
    """
    Safely process gap data to prevent data leakage by ensuring temporal boundaries are respected.
    
    Args:
        df: Raw dataframe with all fetched data (including historical buffer)
        gap_start: Start of gap period
        gap_end: End of gap period
        
    Returns:
        DataFrame with gap data only, processed safely
    """
    print(f"🔒 Safely processing gap data from {gap_start} to {gap_end}")
    
    # Filter to only include data within the gap period (gap_end is exclusive)
    gap_mask = (df['datetime'] >= gap_start) & (df['datetime'] < gap_end)
    gap_df = df[gap_mask].copy()
    
    if gap_df.empty:
        print("⚠️ No data found within gap period")
        return gap_df
    
    # Get historical data before the gap for rolling calculations
    historical_mask = (df['datetime'] < gap_start)
    historical_df = df[historical_mask].copy()
    
    # Sort by datetime to ensure proper temporal order
    gap_df = gap_df.sort_values('datetime').reset_index(drop=True)
    
    # Apply feature engineering with temporal safety
    print("   📊 Calculating engineered features...")
    gap_df['pm2_5_pm10_ratio'] = gap_df.apply(
        lambda row: safe_calculation('pm_ratio', row.get('pm2_5_nowcast'), row.get('pm10_nowcast')), 
        axis=1
    )
    
    gap_df['traffic_index'] = gap_df.apply(
        lambda row: safe_calculation('traffic_index', row.get('no2_ppb'), row.get('co_ppm_8hr_avg')), 
        axis=1
    )
    
    gap_df['total_pm'] = gap_df.apply(
        lambda row: safe_calculation('total_pm', row.get('pm2_5_nowcast'), row.get('pm10_nowcast')), 
        axis=1
    )
    

    
    # Temporal features (safe as they only depend on datetime)
    print("   🕐 Calculating temporal features...")
    gap_df = create_temporal_features(gap_df)
    
    # Safe lag features - use historical data when available
    print("   ⏰ Calculating lag features safely...")
    
    # Get the last values from historical data for the first gap record
    if len(historical_df) > 0:
        last_historical = historical_df.iloc[-1]
        
        # Create pollutant lag features safely
        gap_df['pm2_5_nowcast_lag_1h'] = gap_df['pm2_5_nowcast'].shift(1)
        gap_df['pm10_nowcast_lag_1h'] = gap_df['pm10_nowcast'].shift(1)
        gap_df['co_ppm_8hr_avg_lag_1h'] = gap_df['co_ppm_8hr_avg'].shift(1)
        gap_df['no2_ppb_lag_1h'] = gap_df['no2_ppb'].shift(1)
        
        # Removed AQI lag features to prevent data leakage
        # AQI is calculated from pollutants, so we can't reliably populate AQI lag features
        # during real-time forecasting if pollutant measurements are delayed/unavailable
        
        # Fill the first gap record's lag values with historical data
        gap_df.iloc[0, gap_df.columns.get_loc('pm2_5_nowcast_lag_1h')] = last_historical['pm2_5_nowcast']
        gap_df.iloc[0, gap_df.columns.get_loc('pm10_nowcast_lag_1h')] = last_historical['pm10_nowcast']
        gap_df.iloc[0, gap_df.columns.get_loc('co_ppm_8hr_avg_lag_1h')] = last_historical['co_ppm_8hr_avg']
        gap_df.iloc[0, gap_df.columns.get_loc('no2_ppb_lag_1h')] = last_historical['no2_ppb']
        # Removed AQI lag feature filling since we're not using AQI lag features anymore
    else:
        # No historical data available, use standard shift
        gap_df['pm2_5_nowcast_lag_1h'] = gap_df['pm2_5_nowcast'].shift(1)
        gap_df['pm10_nowcast_lag_1h'] = gap_df['pm10_nowcast'].shift(1)
        gap_df['co_ppm_8hr_avg_lag_1h'] = gap_df['co_ppm_8hr_avg'].shift(1)
        gap_df['no2_ppb_lag_1h'] = gap_df['no2_ppb'].shift(1)
        
        # Removed AQI lag features to prevent data leakage
        # AQI is calculated from pollutants, so we can't reliably populate AQI lag features
        # during real-time forecasting if pollutant measurements are delayed/unavailable
    
    # Safe rolling mean features - calculate using only historical data up to current timestamp
    print("   📊 Calculating 6-hour rolling mean features safely...")
    
    # Initialize rolling mean columns
    gap_df['pm2_5_nowcast_ma_6h'] = np.nan
    gap_df['pm10_nowcast_ma_6h'] = np.nan
    gap_df['no2_ppb_ma_6h'] = np.nan
    
    for idx in range(len(gap_df)):
        current_time = gap_df.iloc[idx]['datetime']
        
        # Combine historical data before gap with gap data up to current time
        historical_up_to_current = pd.concat([
            historical_df[historical_df['datetime'] <= current_time],
            gap_df[gap_df['datetime'] <= current_time]
        ]).sort_values('datetime')
        
        # Calculate rolling means using only historical data up to current time
        if len(historical_up_to_current) > 0:
            pm2_5_recent = historical_up_to_current['pm2_5_nowcast'].tail(6)
            gap_df.iloc[idx, gap_df.columns.get_loc('pm2_5_nowcast_ma_6h')] = pm2_5_recent.mean() if len(pm2_5_recent) > 0 else np.nan
            
            pm10_recent = historical_up_to_current['pm10_nowcast'].tail(6)
            gap_df.iloc[idx, gap_df.columns.get_loc('pm10_nowcast_ma_6h')] = pm10_recent.mean() if len(pm10_recent) > 0 else np.nan
            
            no2_recent = historical_up_to_current['no2_ppb'].tail(6)
            gap_df.iloc[idx, gap_df.columns.get_loc('no2_ppb_ma_6h')] = no2_recent.mean() if len(no2_recent) > 0 else np.nan
    
    # Add datetime_id
    gap_df['datetime_id'] = gap_df['datetime'].astype('int64') // 10**9
    
    print(f"✅ Safely processed {len(gap_df)} gap records")
    return gap_df

def create_lag_features(df: pd.DataFrame, target_col: str, lags: list) -> pd.DataFrame:
    """
    Create lag features for a target column using proper time-based approach.
    This function prevents data leakage by using shift() correctly.
    
    Args:
        df: DataFrame with datetime index
        target_col: Column to create lags for
        lags: List of lag periods (in hours)
    
    Returns:
        DataFrame with lag features added
    """
    df = df.copy()
    
    for lag in lags:
        lag_col = f"{target_col}_lag_{lag}h"
        df[lag_col] = df[target_col].shift(lag)
        
        # Fill NaN values created by lag (first few rows)
        # Use forward fill first, then mean for any remaining NaNs
        df[lag_col] = df[lag_col].fillna(method='ffill')
        if df[lag_col].isna().any():
            df[lag_col] = df[lag_col].fillna(df[target_col].mean())
    
    return df

def apply_all_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps with improved error handling."""
    print("🔧 Applying improved feature engineering...")
    df = df.copy()
    
    # Validate input data
    is_valid, issues = validate_input_data(df)
    if not is_valid:
        print("⚠️  Input data validation issues found:")
        for issue in issues:
            print(f"   - {issue}")
    
    # Apply safe engineered features
    print("   📊 Calculating engineered features...")
    df['pm2_5_pm10_ratio'] = df.apply(
        lambda row: safe_calculation('pm_ratio', row.get('pm2_5_nowcast'), row.get('pm10_nowcast')), 
        axis=1
    )
    
    df['traffic_index'] = df.apply(
        lambda row: safe_calculation('traffic_index', row.get('no2_ppb'), row.get('co_ppm_8hr_avg')), 
        axis=1
    )
    
    df['total_pm'] = df.apply(
        lambda row: safe_calculation('total_pm', row.get('pm2_5_nowcast'), row.get('pm10_nowcast')), 
        axis=1
    )
    

    
    # Temporal features
    print("   🕐 Calculating temporal features...")
    df = create_temporal_features(df)
    
    # Create lag features using proper time-based approach
    print("   ⏰ Creating lag features for pollutant concentrations...")
    df = create_lag_features(df, 'pm2_5_nowcast', [1])
    df = create_lag_features(df, 'pm10_nowcast', [1])
    df = create_lag_features(df, 'co_ppm_8hr_avg', [1])
    df = create_lag_features(df, 'no2_ppb', [1])
    
    # Removed AQI lag features to prevent data leakage
    # AQI is calculated from pollutants, so using AQI lag features would require
    # pollutant measurements to be available, which may not be the case in real-time
    
    # Rolling mean features (6-hour window)
    print("   📊 Calculating 6-hour rolling mean features...")
    df['pm2_5_nowcast_ma_6h'] = df['pm2_5_nowcast'].rolling(window=6, min_periods=1).mean()
    df['pm10_nowcast_ma_6h'] = df['pm10_nowcast'].rolling(window=6, min_periods=1).mean()
    df['no2_ppb_ma_6h'] = df['no2_ppb'].rolling(window=6, min_periods=1).mean()
    
    # Add datetime_id as Unix timestamp for online compatibility
    df['datetime_id'] = df['datetime'].astype('int64') // 10**9
    
    # Report feature engineering results
    engineered_features = ['pm2_5_pm10_ratio', 'traffic_index']
    cyclical_features = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']
    lag_features = ['pm2_5_nowcast_lag_1h', 'pm10_nowcast_lag_1h', 'co_ppm_8hr_avg_lag_1h', 'no2_ppb_lag_1h']
                   # Removed AQI lag features to prevent data leakage - AQI is calculated from pollutants
    rolling_features = ['pm2_5_nowcast_ma_6h', 'pm10_nowcast_ma_6h', 'no2_ppb_ma_6h']
    
    print("   📈 Feature engineering results:")
    
    # Engineered features
    print("      📊 Engineered features:")
    for feature in engineered_features:
        if feature in df.columns:
            missing_count = df[feature].isnull().sum()
            total_count = len(df)
            valid_count = total_count - missing_count
            print(f"         {feature}: {valid_count}/{total_count} valid values ({missing_count} missing)")
    
    # Cyclical features
    print("      🔄 Cyclical features:")
    for feature in cyclical_features:
        if feature in df.columns:
            missing_count = df[feature].isnull().sum()
            total_count = len(df)
            valid_count = total_count - missing_count
            print(f"         {feature}: {valid_count}/{total_count} valid values ({missing_count} missing)")
    
    # Lag features
    print("      ⏰ Lag features:")
    for feature in lag_features:
        if feature in df.columns:
            missing_count = df[feature].isnull().sum()
            total_count = len(df)
            valid_count = total_count - missing_count
            print(f"         {feature}: {valid_count}/{total_count} valid values ({missing_count} missing)")
    
    # Rolling features
    print("      📊 Rolling mean features:")
    for feature in rolling_features:
        if feature in df.columns:
            missing_count = df[feature].isnull().sum()
            total_count = len(df)
            valid_count = total_count - missing_count
            print(f"         {feature}: {valid_count}/{total_count} valid values ({missing_count} missing)")
    
    # Select only production features, handling missing columns gracefully
    feature_cols = list(PRODUCTION_FEATURES_SCHEMA.keys())
    
    # Check for missing columns and add them with default values if needed
    missing_cols = []
    for col in feature_cols:
        if col not in df.columns:
            missing_cols.append(col)
            if col == 'aqi_epa_calc':
                df[col] = np.nan  # Will be calculated elsewhere in the pipeline
            elif col == 'so2_ppb':
                df[col] = np.nan  # Not available in test data
            else:
                df[col] = np.nan  # Default for any other missing columns
    
    if missing_cols:
        print(f"⚠️  Added missing columns with default values: {missing_cols}")
    
    df_features = df[feature_cols].copy()
    
    print(f"✅ Improved feature engineering complete. Shape: {df_features.shape}")
    return df_features

# Duplicate functions removed - using consolidated apply_all_feature_engineering function above

# ============================================================
# Data Quality and Imputation Functions  
# ============================================================
# ============================================================
# Gap Detection and Filling Functions
# ============================================================

def get_last_record_timestamp(fs) -> Optional[datetime]:
    """
    Get the timestamp of the last record in the Hopsworks feature group.
    
    Args:
        fs: Hopsworks feature store
        
    Returns:
        datetime: Timestamp of the last record in PKT, or None if no records exist
    """
    try:
        # Get the latest version of the feature group
        latest_version = get_latest_feature_group_version(fs, FEATURE_GROUP_NAME)
        if latest_version == 0:
            print("📋 No existing feature group found")
            return None
        
        fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=latest_version)
        if fg is None:
            print("📋 Feature group not accessible")
            return None
        
        # Query for the latest record by datetime
        query = fg.select(['datetime'])
        df_all = query.read()
        
        if df_all.empty:
            print("📋 No records found in feature group")
            return None
        
        # Sort by datetime to get the latest record
        df_all['datetime'] = pd.to_datetime(df_all['datetime'])
        df_sorted = df_all.sort_values('datetime', ascending=False)
        last_timestamp = df_sorted['datetime'].iloc[0]
        
        # Convert UTC to PKT for comparison
        if last_timestamp.tz is None:
            # If timezone-naive, assume it's already in PKT
            last_timestamp_pkt = last_timestamp
        else:
            # If timezone-aware, convert to PKT
            last_timestamp_pkt = last_timestamp.astimezone(PKT).replace(tzinfo=None)
        
        print(f"📅 Last record in Hopsworks: {last_timestamp_pkt} (PKT)")
        return last_timestamp_pkt
        
    except Exception as e:
        print(f"⚠️ Error getting last record timestamp: {str(e)}")
        return None

def detect_missing_hours(fs) -> Tuple[Optional[datetime], Optional[datetime], int]:
    """
    Detect missing hours between the last record and current time.
    
    Args:
        fs: Hopsworks feature store
        
    Returns:
        Tuple[start_time, end_time, gap_hours]: 
        - start_time: When to start fetching (PKT)
        - end_time: When to end fetching (PKT) 
        - gap_hours: Number of missing hours
    """
    try:
        # Get last record timestamp
        last_timestamp = get_last_record_timestamp(fs)
        
        # Current time in PKT (timezone-naive to match last_timestamp)
        current_time_pkt = datetime.now(PKT).replace(tzinfo=None)
        current_hour = current_time_pkt.replace(minute=0, second=0, microsecond=0)
        
        if last_timestamp is None:
            print("⚠️ No existing records found - this should trigger backfill instead")
            return None, None, 0
        
        # Ensure last_timestamp is timezone-naive for comparison
        if hasattr(last_timestamp, 'tz') and last_timestamp.tz is not None:
            last_timestamp = last_timestamp.replace(tzinfo=None)
        
        # Calculate the next hour after the last record
        next_expected_hour = (last_timestamp + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        
        # Calculate gap
        if next_expected_hour >= current_hour:
            print("✅ No gap detected - data is up to date")
            return None, None, 0
        
        gap_hours = int((current_hour - next_expected_hour).total_seconds() / 3600)
        
        print(f"🔍 Gap detected:")
        print(f"   📅 Last record: {last_timestamp}")
        print(f"   📅 Next expected: {next_expected_hour}")
        print(f"   📅 Current hour: {current_hour}")
        print(f"   ⏰ Missing hours: {gap_hours}")
        
        return next_expected_hour, current_hour, gap_hours
        
    except Exception as e:
        print(f"❌ Error detecting missing hours: {str(e)}")
        return None, None, 0

def fetch_and_process_gap_data(fs, gap_start: datetime, gap_end: datetime, gap_hours: int) -> pd.DataFrame:
    """
    Fetch and process data for the detected gap period with robust error handling.
    
    Args:
        fs: Hopsworks feature store
        gap_start: Start datetime of the gap
        gap_end: End datetime of the gap  
        gap_hours: Number of hours in the gap
        
    Returns:
        DataFrame with processed gap data or None if failed
    """
    print(f"🔧 Fetching gap data from {gap_start} to {gap_end} ({gap_hours} hours)")
    
    try:
        # For large gaps, process in chunks to avoid API timeouts
        max_chunk_hours = 48  # Process max 48 hours at a time
        all_gap_data = []
        
        current_start = gap_start
        chunks_processed = 0
        max_retries_per_chunk = 3
        
        while current_start < gap_end:
            # Calculate chunk end (don't exceed gap_end)
            chunk_hours = min(max_chunk_hours, (gap_end - current_start).total_seconds() / 3600)
            current_end = current_start + timedelta(hours=chunk_hours)
            
            if current_end > gap_end:
                current_end = gap_end
            
            chunk_hours_actual = int((current_end - current_start).total_seconds() / 3600)
            print(f"   📦 Processing chunk {chunks_processed + 1}: {current_start} to {current_end} ({chunk_hours_actual}h)")
            
            # Retry mechanism for each chunk
            chunk_success = False
            for retry in range(max_retries_per_chunk):
                try:
                    # Fetch data for this chunk with extended time range for proper averages
                    # Add 16 hours before start for proper average calculation
                    fetch_start = current_start - timedelta(hours=16)
                    
                    json_data = fetch_aqi_data_with_retry(LAT, LON, fetch_start, current_end)
                    if not json_data:
                        if retry < max_retries_per_chunk - 1:
                            print(f"      ⚠️ Chunk fetch failed, retry {retry + 1}/{max_retries_per_chunk}")
                            time.sleep(5 * (retry + 1))  # Exponential backoff
                            continue
                        else:
                            print(f"      ❌ Chunk fetch failed after {max_retries_per_chunk} retries")
                            break
                    
                    # Process chunk data
                    df_chunk = json_to_dataframe(json_data)
                    if df_chunk is None or df_chunk.empty:
                        if retry < max_retries_per_chunk - 1:
                            print(f"      ⚠️ Empty chunk data, retry {retry + 1}/{max_retries_per_chunk}")
                            time.sleep(3 * (retry + 1))
                            continue
                        else:
                            print(f"      ❌ Empty chunk data after {max_retries_per_chunk} retries")
                            break
                    
                    # Apply SAFE feature engineering for gap data
                    df_chunk_features = safe_process_gap_data(df_chunk, current_start, current_end)
                    
                    # Filter to only the gap period (exclude the 16-hour buffer)
                    df_chunk_gap = df_chunk_features[
                        (df_chunk_features['datetime'] >= current_start) & 
                        (df_chunk_features['datetime'] < current_end)
                    ].copy()
                    feature_cols = list(PRODUCTION_FEATURES_SCHEMA.keys())
                    for col in feature_cols:
                        if col not in df_chunk_gap.columns:
                            df_chunk_gap[col] = np.nan
                    df_chunk_gap = df_chunk_gap[feature_cols]
                    
                    if not df_chunk_gap.empty:
                        all_gap_data.append(df_chunk_gap)
                        print(f"      ✅ Chunk processed: {len(df_chunk_gap)} records")
                        chunk_success = True
                        break
                    else:
                        if retry < max_retries_per_chunk - 1:
                            print(f"      ⚠️ No gap data in chunk, retry {retry + 1}/{max_retries_per_chunk}")
                            time.sleep(2 * (retry + 1))
                            continue
                        else:
                            print(f"      ❌ No gap data in chunk after {max_retries_per_chunk} retries")
                            break
                            
                except Exception as e:
                    if retry < max_retries_per_chunk - 1:
                        print(f"      ⚠️ Chunk error: {str(e)}, retry {retry + 1}/{max_retries_per_chunk}")
                        time.sleep(5 * (retry + 1))
                        continue
                    else:
                        print(f"      ❌ Chunk failed after {max_retries_per_chunk} retries: {str(e)}")
                        break
            
            if not chunk_success:
                print(f"   ❌ Failed to process chunk {chunks_processed + 1}, continuing with next chunk...")
                # Continue with next chunk instead of failing completely
            
            # Move to next chunk
            current_start = current_end
            chunks_processed += 1
            
            # Add small delay between chunks to be respectful to API
            if current_start < gap_end:
                time.sleep(2)
        
        # Combine all chunk data
        if all_gap_data:
            gap_data = pd.concat(all_gap_data, ignore_index=True)
            gap_data = gap_data.sort_values('datetime').drop_duplicates(subset=['datetime'])
            
            print(f"✅ Gap data processing completed:")
            print(f"   📊 Total records: {len(gap_data)}")
            print(f"   📅 Date range: {gap_data['datetime'].min()} to {gap_data['datetime'].max()}")
            print(f"   📦 Chunks processed: {chunks_processed}")
            
            return gap_data
        else:
            print("❌ No gap data could be retrieved from any chunk")
            return None
            
    except Exception as e:
        print(f"❌ Critical error in gap data processing: {str(e)}")
        return None

def connect_to_hopsworks() -> Tuple[Any, Any]:
    """Connect to Hopsworks and return project and feature store."""
    if not HOPSWORKS_AVAILABLE:
        raise ValueError("❌ Hopsworks credentials not available. Cannot connect.")
    
    try:
        print("🔗 Connecting to Hopsworks...")
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT_NAME
        )
        fs = project.get_feature_store()
        print("✅ Connected to Hopsworks successfully")
        return project, fs
    except Exception as e:
        print(f"❌ Failed to connect to Hopsworks: {str(e)}")
        raise

def get_historical_data_for_imputation(fs, feature_name: str, hours_back: int = 168) -> pd.Series:
    """Get historical data from Hopsworks for imputation (last 7 days = 168 hours)."""
    try:
        # Use the latest available version
        latest_version = get_latest_feature_group_version(fs, FEATURE_GROUP_NAME)
        if latest_version == 0:
            return pd.Series(dtype=float)
        
        fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=latest_version)
        if fg is None:
            return pd.Series(dtype=float)
        
        end_time = datetime.now(PKT)
        start_time = end_time - timedelta(hours=hours_back)
        
        query = fg.select([feature_name]).filter(
            (fg.datetime >= start_time) & (fg.datetime <= end_time)
        )
        
        df_hist = query.read()
        if not df_hist.empty and feature_name in df_hist.columns:
            return df_hist[feature_name].dropna()
        else:
            return pd.Series(dtype=float)
            
    except Exception as e:
        print(f"⚠️ Could not fetch historical data for {feature_name}: {str(e)}")
        return pd.Series(dtype=float)

def impute_missing_values(df: pd.DataFrame, fs) -> pd.DataFrame:
    """Impute missing values using historical Hopsworks data."""
    df = df.copy()
    print("🔧 Applying imputation for missing values...")
    
    pollutant_features = ['pm2_5_nowcast', 'pm10_nowcast', 'co_ppm_8hr_avg', 'no2_ppb', 'so2_ppb']
    
    for feature in pollutant_features:
        if feature in df.columns:
            missing_mask = df[feature].isna() | (df[feature] == 0)
            if missing_mask.any():
                print(f"  🔧 Imputing {missing_mask.sum()} missing values for {feature}")
                
                # Try to get historical data for imputation
                hist_data = get_historical_data_for_imputation(fs, feature)
                
                if len(hist_data) > 0:
                    impute_value = hist_data.median()
                    print(f"    ✅ Using historical median: {impute_value:.2f} (from {len(hist_data)} historical records)")
                    df.loc[missing_mask, feature] = impute_value
                else:
                    print(f"    ⚠️ No historical data available, using forward/backward fill")
                    # Use modern pandas methods instead of deprecated fillna(method=...)
                    df[feature] = df[feature].ffill().bfill()
                    
                    # Check if imputation was successful
                    remaining_missing = df[feature].isna().sum()
                    if remaining_missing > 0:
                        print(f"    ❌ Warning: {remaining_missing} values still missing after imputation")
                    else:
                        print(f"    ✅ Successfully imputed all missing values")
            else:
                print(f"  ✅ No missing values for {feature}")
    
    return df

def validate_data_quality(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate data quality before uploading to Hopsworks."""
    print("🔍 Validating data quality...")
    
    core_features = ['pm2_5_nowcast', 'pm10_nowcast', 'co_ppm_8hr_avg', 'no2_ppb', 'so2_ppb']
    
    for feature in core_features:
        if feature in df.columns:
            missing_pct = df[feature].isna().sum() / len(df)
            if missing_pct > MISSING_DATA_THRESHOLD:
                return False, f"❌ {feature} has {missing_pct:.1%} missing data (threshold: {MISSING_DATA_THRESHOLD:.1%})"
    
    if 'aqi_epa_calc' in df.columns:
        target_missing_pct = df['aqi_epa_calc'].isna().sum() / len(df)
        if target_missing_pct > 0.5:
            return False, f"❌ Target variable has {target_missing_pct:.1%} missing data"
    
    print("✅ Data quality validation passed")
    return True, "Data quality validation passed"

# ============================================================
# Hopsworks Integration Functions
# ============================================================

def get_latest_feature_group_version(fs, feature_group_name: str) -> int:
    """
    Get the latest version of a feature group from Hopsworks.
    
    Args:
        fs: Hopsworks feature store
        feature_group_name: Name of the feature group
        
    Returns:
        int: Latest version number, or 0 if no versions exist
    """
    try:
        # Get all feature groups with the given name
        feature_groups = fs.get_feature_groups(name=feature_group_name)
        
        if not feature_groups:
            print(f"📋 No existing feature groups found for '{feature_group_name}'")
            return 0
        
        # Extract version numbers and find the maximum
        versions = [fg.version for fg in feature_groups]
        latest_version = max(versions)
        
        print(f"📊 Found existing versions for '{feature_group_name}': {sorted(versions)}")
        print(f"🔢 Latest version: {latest_version}")
        
        return latest_version
        
    except Exception as e:
        print(f"⚠️ Error getting feature group versions: {str(e)}")
        print(f"🔄 Defaulting to version 0")
        return 0

def determine_next_version(fs, feature_group_name: str, df_sample: pd.DataFrame, force_new_version: bool = False) -> int:
    """
    Determine the next version number for a feature group.
    
    Args:
        fs: Hopsworks feature store
        feature_group_name: Name of the feature group
        df_sample: Sample dataframe to check schema compatibility
        force_new_version: If True, always create a new version
        
    Returns:
        int: Next version number to use
    """
    try:
        latest_version = get_latest_feature_group_version(fs, feature_group_name)
        
        if latest_version == 0:
            print(f"🆕 Creating first version of feature group '{feature_group_name}'")
            return 1
        
        if force_new_version:
            next_version = latest_version + 1
            print(f"🔄 Force creating new version: {next_version}")
            return next_version
        
        # Check if we can use the existing latest version
        try:
            existing_fg = fs.get_feature_group(name=feature_group_name, version=latest_version)
            
            # Compare schemas (column names and types)
            existing_schema = {f.name: f.type for f in existing_fg.features}
            new_schema = {col: str(df_sample[col].dtype) for col in df_sample.columns}
            
            if existing_schema.keys() == new_schema.keys():
                print(f"✅ Schema compatible with existing version {latest_version}")
                return latest_version
            else:
                next_version = latest_version + 1
                print(f"🔄 Schema changed, creating new version: {next_version}")
                print(f"   📋 New columns: {set(new_schema.keys()) - set(existing_schema.keys())}")
                print(f"   📋 Removed columns: {set(existing_schema.keys()) - set(new_schema.keys())}")
                return next_version
                
        except Exception as e:
            # If we can't access the existing feature group, create a new version
            next_version = latest_version + 1
            print(f"⚠️ Cannot access existing version {latest_version}: {str(e)}")
            print(f"🔄 Creating new version: {next_version}")
            return next_version
            
    except Exception as e:
        print(f"❌ Error determining next version: {str(e)}")
        print(f"🔄 Defaulting to version 1")
        return 1

def create_or_get_feature_group(fs, df_sample: pd.DataFrame, force_new_version: bool = False):
    """Create or get existing feature group with automated versioning."""
    try:
        # Determine the appropriate version to use
        version_to_use = determine_next_version(fs, FEATURE_GROUP_NAME, df_sample, force_new_version)
        
        print(f"🔧 Creating or getting feature group: {FEATURE_GROUP_NAME} v{version_to_use}")
        
        # Use the simple get_or_create approach that works in enhanced pipeline
        fg = fs.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=version_to_use,
            description="Karachi AQI features with EPA calculations and engineered features",
            primary_key=["datetime"],  # Use datetime as primary key (like enhanced pipeline)
            event_time="datetime"
        )
        
        print(f"✅ Successfully got/created feature group: {FEATURE_GROUP_NAME} v{version_to_use}")
        return fg
        
    except Exception as e:
        print(f"❌ Failed to create or get feature group: {str(e)}")
        return None

def upload_to_hopsworks(df: pd.DataFrame, fs) -> bool:
    """Upload data to Hopsworks feature group with batch processing for large datasets."""
    try:
        print(f"📤 Uploading {len(df)} records to Hopsworks...")
        feature_cols = list(PRODUCTION_FEATURES_SCHEMA.keys())
        for col in feature_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[feature_cols]

        # Get or create feature group
        fg = create_or_get_feature_group(fs, df)
        
        if fg is None:
            print("❌ Failed to get or create feature group")
            return False
        
        # Ensure datetime is timezone-aware
        if df['datetime'].dt.tz is None:
            df['datetime'] = df['datetime'].dt.tz_localize(PKT)
        
        # For large datasets, upload in batches to avoid Kafka timeout
        batch_size = 1000  # Smaller batch size to avoid timeout
        total_records = len(df)
        successful_batches = 0
        failed_batches = 0
        
        if total_records <= batch_size:
            # Small dataset - upload directly
            print(f"📤 Uploading {total_records} records directly...")
            fg.insert(df, write_options={"wait_for_job": True})
            successful_batches = 1
        else:
            # Large dataset - upload in batches
            print(f"📤 Uploading {total_records} records in batches of {batch_size}...")
            
            for i in range(0, total_records, batch_size):
                batch_end = min(i + batch_size, total_records)
                batch_df = df.iloc[i:batch_end].copy()
                
                print(f"   📦 Batch {i//batch_size + 1}: records {i+1}-{batch_end}")
                
                # Add delay between batches to avoid overwhelming Kafka
                if i > 0:
                    time.sleep(2)
                
                try:
                    fg.insert(batch_df, write_options={"wait_for_job": True})
                    print(f"   ✅ Batch {i//batch_size + 1} uploaded successfully")
                    successful_batches += 1
                except Exception as batch_error:
                    print(f"   ❌ Batch {i//batch_size + 1} failed: {str(batch_error)}")
                    failed_batches += 1
                    # Continue with next batch but track the failure
                    continue
            
            print(f"📊 Batch upload summary: {successful_batches} successful, {failed_batches} failed")
            
            # If all batches failed, return failure
            if successful_batches == 0:
                print("❌ All batches failed - upload unsuccessful")
                return False
        
        # Only show success if we actually uploaded data
        if successful_batches > 0:
            print(f"✅ Successfully uploaded {len(df)} records to Hopsworks")
            print(f"   📊 Feature Group: {FEATURE_GROUP_NAME}")
            print(f"   📈 Features: {len(df.columns) - 1}")  # Exclude datetime
            print(f"   📅 Time Range: {df['datetime'].min()} to {df['datetime'].max()}")
            return True
        else:
            print("❌ No data was successfully uploaded to Hopsworks")
            return False
        
    except Exception as e:
        print(f"❌ Failed to upload to Hopsworks: {str(e)}")
        print(f"🔍 Error details: {type(e).__name__}")
        return False

def check_feature_group_exists(fs, version: Optional[int] = None) -> bool:
    """Check if the feature group already exists with automated version detection."""
    try:
        if version is None:
            # Check if any version exists
            latest_version = get_latest_feature_group_version(fs, FEATURE_GROUP_NAME)
            if latest_version == 0:
                return False
            version = latest_version
        
        fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=version)
        return fg is not None
    except Exception:
        return False

def check_existing_data_coverage(fs, start_date: datetime, end_date: datetime, version: Optional[int] = None) -> Tuple[bool, int, str]:
    """
    Check if feature group already contains data for the specified date range.
    
    Returns:
        Tuple[bool, int, str]: (has_sufficient_data, record_count, message)
    """
    try:
        # Determine version to use
        if version is None:
            version = get_latest_feature_group_version(fs, FEATURE_GROUP_NAME)
            if version == 0:
                return False, 0, "No feature group versions exist"
        
        # Get feature group
        fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=version)
        if fg is None:
            return False, 0, f"Feature group v{version} doesn't exist"
        
        # Query data for the date range
        query = fg.select_all().filter(
            (fg.datetime >= start_date) & (fg.datetime <= end_date)
        )
        
        # Get count of existing records
        df_existing = query.read()
        record_count = len(df_existing)
        
        # Calculate expected records (approximately hourly data)
        expected_hours = int((end_date - start_date).total_seconds() / 3600)
        coverage_ratio = record_count / expected_hours if expected_hours > 0 else 0
        
        # Consider data sufficient if we have at least 80% coverage
        has_sufficient_data = coverage_ratio >= 0.8
        
        if has_sufficient_data:
            date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            message = f"Sufficient data exists ({record_count} records, {coverage_ratio:.1%} coverage) for {date_range} in v{version}"
        else:
            message = f"Insufficient data coverage ({coverage_ratio:.1%}) for the requested period in v{version}"
        
        return has_sufficient_data, record_count, message
        
    except Exception as e:
        return False, 0, f"Error checking existing data: {str(e)}"

# ============================================================
# Pipeline Functions
# ============================================================
def backfill_pipeline(force: bool = False) -> bool:
    """
    Backfill pipeline: Fetch 12 months of historical data and create feature group.
    
    Args:
        force: If True, re-process data even if it already exists
    """
    print("\n" + "="*60)
    print("🚀 STARTING BACKFILL PIPELINE (12 MONTHS)")
    print("="*60)
    
    try:
        # Connect to Hopsworks
        project, fs = connect_to_hopsworks()
        
        # Set date range for 12 months
        end_date = datetime.now(timezone.utc)
        start_date = end_date - relativedelta(months=13)
        
        print(f"📅 Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Check if data already exists (unless force is True)
        if not force:
            has_data, record_count, message = check_existing_data_coverage(fs, start_date, end_date)
            if has_data:
                print(f"✅ {message}")
                print("💡 Use --force flag to re-process existing data")
                print("✅ BACKFILL PIPELINE SKIPPED - DATA ALREADY EXISTS")
                return True
            else:
                print(f"ℹ️ {message}")
                print("📥 Proceeding with data fetch and processing...")
        else:
            print("🔄 Force mode enabled - will re-process data even if it exists")
        
        # Fetch data with retry
        json_data = fetch_aqi_data_with_retry(LAT, LON, start_date, end_date)
        if not json_data:
            print("❌ Failed to fetch data for backfill")
            return False
        
        # Process data
        df = json_to_dataframe(json_data)
        if df is None or df.empty:
            print("❌ No data available for processing")
            return False
        
        # Apply feature engineering
        df_features = apply_all_feature_engineering(df)
        
        # For backfill: Drop initial rows with empty NowCast values
        # NowCast requires historical data, so initial rows will have NaN values
        print("🧹 Dropping initial rows with empty NowCast values (backfill mode)...")
        initial_count = len(df_features)
        
        # Drop rows where both PM2.5 and PM10 NowCast values are missing
        nowcast_mask = df_features['pm2_5_nowcast'].notna() & df_features['pm10_nowcast'].notna()
        df_features = df_features[nowcast_mask].copy()
        
        dropped_count = initial_count - len(df_features)
        print(f"📊 Dropped {dropped_count} initial rows with empty NowCast values")
        print(f"📊 Remaining records: {len(df_features)}")
        
        # Remove rows where target is missing
        df_features = df_features.dropna(subset=['aqi_epa_calc'])
        
        # Impute missing values (will use forward/backward fill for backfill)
        df_features = impute_missing_values(df_features, fs)
        
        # Validate data quality
        is_valid, message = validate_data_quality(df_features)
        if not is_valid:
            print(f"❌ Data quality validation failed: {message}")
            return False
        
        # Upload to Hopsworks (overwrite mode for backfill)
        success = upload_to_hopsworks(df_features, fs)
        
        if success:
            print(f"\n✅ BACKFILL PIPELINE COMPLETED SUCCESSFULLY")
            print(f"📊 Processed {len(df_features)} records")
            print(f"🎯 Features: {len(PRODUCTION_FEATURES_SCHEMA)} total")
            return True
        else:
            print("❌ BACKFILL PIPELINE FAILED")
            return False
            
    except Exception as e:
        print(f"❌ BACKFILL PIPELINE ERROR: {str(e)}")
        return False

def hourly_pipeline() -> bool:
    """
    Enhanced hourly pipeline: Detect and fill data gaps, then fetch recent data.
    If feature group doesn't exist, run backfill first.
    """
    print("\n" + "="*60)
    print("⏰ STARTING ENHANCED HOURLY PIPELINE WITH GAP DETECTION")
    print("="*60)
    
    try:
        # Connect to Hopsworks
        project, fs = connect_to_hopsworks()
        
        # Check if feature group exists
        if not check_feature_group_exists(fs):
            print("⚠️ Feature group doesn't exist. Running backfill first...")
            backfill_success = backfill_pipeline()
            if not backfill_success:
                print("❌ Backfill failed, cannot proceed with hourly pipeline")
                return False
        
        # Step 1: Detect missing hours (gaps)
        print("🔍 Step 1: Detecting data gaps...")
        gap_start, gap_end, gap_hours = detect_missing_hours(fs)
        
        # Step 2: Fill gaps if detected
        if gap_hours > 0:
            print(f"🔧 Step 2: Filling {gap_hours} missing hours...")
            
            # Only fill gaps if they're reasonable (not more than 7 days = 168 hours)
            if gap_hours > 168:
                print(f"⚠️ Gap too large ({gap_hours} hours > 168 hours). Consider running backfill instead.")
                print("💡 Proceeding with normal hourly update for current hour only.")
                gap_hours = 0  # Skip gap filling
            else:
                gap_data = fetch_and_process_gap_data(fs, gap_start, gap_end, gap_hours)
                
                if gap_data is not None and not gap_data.empty:
                    # Validate gap data quality
                    is_valid, message = validate_data_quality(gap_data)
                    if is_valid:
                        # Upload gap data
                        gap_success = upload_to_hopsworks(gap_data, fs)
                        if gap_success:
                            print(f"✅ Successfully filled {len(gap_data)} missing records")
                            print(f"   📅 Gap filled from: {gap_data['datetime'].min()}")
                            print(f"   📅 Gap filled to: {gap_data['datetime'].max()}")
                        else:
                            print("❌ Failed to upload gap data")
                            return False
                    else:
                        print(f"❌ Gap data quality validation failed: {message}")
                        return False
                else:
                    print("⚠️ Could not fetch gap data, proceeding with current hour only")
        else:
            print("✅ No gaps detected, proceeding with current hour update")
        
        # Step 3: Fetch and process current hour data (normal hourly pipeline logic)
        print("📡 Step 3: Fetching current hour data...")
        
        # Fetch last 19 hours: 16 hours for proper averages + 3 hours for lag calculation
        # This ensures CO 8hr avg and PM NowCast have sufficient data
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(hours=19)
        
        print(f"📅 Fetching last 19 hours: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
        print("   (16 hours for proper averages + 3 hours for lag calculation)")
        
        # Fetch data with retry
        json_data = fetch_aqi_data_with_retry(LAT, LON, start_date, end_date)
        if not json_data:
            print("❌ Failed to fetch current hour data")
            return False
        
        # Process data
        df = json_to_dataframe(json_data)
        if df is None or df.empty:
            print("❌ No data available for processing")
            return False
        
        print(f"📊 Processing {len(df)} records for feature engineering...")
        
        # Apply feature engineering to all data (needed for proper averages)
        df_features = apply_all_feature_engineering(df)
        
        # Skip the first 16 hours where averages may be incomplete
        # Only consider records from hour 17 onwards (index 16+)
        if len(df_features) > 16:
            df_valid = df_features.iloc[16:].copy()
            print(f"⏭️ Skipping first 16 hours with incomplete averages")
            print(f"📊 Considering {len(df_valid)} records with proper averages")
        else:
            print("⚠️ Not enough data for proper average calculation (need >16 hours)")
            df_valid = df_features.copy()
        
        # For hourly updates, only keep the most recent hour with valid target
        df_valid = df_valid.dropna(subset=['aqi_epa_calc'])
        if df_valid.empty:
            print("⚠️ No valid records with AQI target found after skipping incomplete averages")
            return False
        
        # Keep only the latest record (most recent hour with complete averages)
        df_latest = df_valid.tail(1).copy()
        print(f"✅ Selected latest record: {df_latest['datetime'].iloc[0]}")
        
        # Check if this record already exists in Hopsworks (avoid duplicates)
        latest_timestamp = df_latest['datetime'].iloc[0]
        last_record_timestamp = get_last_record_timestamp(fs)
        
        if last_record_timestamp is not None and latest_timestamp <= last_record_timestamp:
            print(f"ℹ️ Latest record ({latest_timestamp}) already exists in Hopsworks")
            print(f"   📅 Last record in Hopsworks: {last_record_timestamp}")
            print("✅ HOURLY PIPELINE COMPLETED - NO NEW DATA TO ADD")
            return True
        
        # Impute missing values using Hopsworks historical data
        df_latest = impute_missing_values(df_latest, fs)
        
        # Validate data quality
        is_valid, message = validate_data_quality(df_latest)
        if not is_valid:
            print(f"❌ Data quality validation failed: {message}")
            return False
        
        # Upload to Hopsworks
        success = upload_to_hopsworks(df_latest, fs)
        
        if success:
            total_records_added = (len(gap_data) if gap_hours > 0 and 'gap_data' in locals() and gap_data is not None else 0) + len(df_latest)
            print(f"\n✅ ENHANCED HOURLY PIPELINE COMPLETED SUCCESSFULLY")
            print(f"📊 Total records added: {total_records_added}")
            if gap_hours > 0:
                print(f"   🔧 Gap records: {len(gap_data) if 'gap_data' in locals() and gap_data is not None else 0}")
            print(f"   📡 Current hour: {len(df_latest)}")
            print(f"🕐 Latest timestamp: {df_latest['datetime'].iloc[0]}")
            return True
        else:
            print("❌ HOURLY PIPELINE FAILED")
            return False
            
    except Exception as e:
        print(f"❌ HOURLY PIPELINE ERROR: {str(e)}")
        return False

# ============================================================
# Display Functions
# ============================================================
def display_table_rich(df: pd.DataFrame, last_n: Optional[int] = None):
    """Display DataFrame using Rich table formatting."""
    console = Console(force_terminal=True, width=200)
    table = Table(show_header=True, header_style="bold cyan", expand=True)
    
    for col in df.columns:
        table.add_column(col, justify="right" if pd.api.types.is_numeric_dtype(df[col]) else "left")
    
    subset = df.tail(last_n) if last_n else df
    for _, row in subset.iterrows():
        values = ["-" if pd.isna(v) else f"{v:.2f}" if isinstance(v, float) else str(v) for v in row]
        table.add_row(*values)
    
    console.print(table)

# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Parse command line arguments
        args = sys.argv[1:]
        mode = args[0].lower()
        force = "--force" in args
        
        if mode == "backfill":
            success = backfill_pipeline(force=force)
        elif mode == "hourly":
            success = hourly_pipeline()
        else:
            print("❌ Invalid mode. Use 'backfill' or 'hourly'")
            print("Usage: python unified_aqi_hopsworks_pipeline.py [backfill|hourly] [--force]")
            sys.exit(1)
        
        sys.exit(0 if success else 1)
    else:
        print("🔧 UNIFIED AQI HOPSWORKS PIPELINE")
        print("=" * 50)
        print("Usage: python unified_aqi_hopsworks_pipeline.py [mode] [options]")
        print("\nModes:")
        print("  backfill  - Initialize with 12 months of historical data")
        print("  hourly    - Add latest hourly data (runs backfill if needed)")
        print("\nOptions:")
        print("  --force   - Force re-processing even if data already exists (backfill only)")
        print("\nFeatures:")
        print(f"  📊 {len(PRODUCTION_FEATURES_SCHEMA)} total features")
        print(f"  🎯 Target: aqi_epa_calc")
        print(f"  🏪 Feature Group: {FEATURE_GROUP_NAME}")
        print(f"  🌍 Location: Karachi ({LAT}, {LON})")


