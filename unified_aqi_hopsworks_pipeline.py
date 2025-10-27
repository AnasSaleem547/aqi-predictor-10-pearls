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
import hopsworks
from typing import Optional, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

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
FEATURE_GROUP_NAME = "karachifeatures"
FEATURE_GROUP_VERSION = 3  # Increment version to bypass corrupted metadata

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
    
    # Engineered features (4 features)
    "pm2_5_pm10_ratio": "float",
    "traffic_index": "float",
    "total_pm": "float",
    "pm_weighted": "float",
    
    # Temporal features (4 features)
    "hour": "int",
    "is_rush_hour": "int",
    "is_weekend": "int", 
    "season": "int",
    
    # Lag features (2 features)
    "pm2_5_nowcast_lag_1h": "float",
    "no2_ppb_lag_1h": "float"
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

    return df

# ============================================================
# Feature Engineering Functions
# ============================================================
def apply_all_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in one function."""
    print("🔧 Applying feature engineering...")
    df = df.copy()
    
    # Engineered ratio and composite features
    df['pm2_5_pm10_ratio'] = df['pm2_5_nowcast'] / (df['pm10_nowcast'] + 1e-6)
    df['traffic_index'] = (df['no2_ppb'] * 0.6) + (df['co_ppm_8hr_avg'] * 0.4)
    df['total_pm'] = df['pm2_5_nowcast'] + df['pm10_nowcast']
    df['pm_weighted'] = (df['pm2_5_nowcast'] * 0.7) + (df['pm10_nowcast'] * 0.3)
    
    # Temporal features
    df['hour'] = df['datetime'].dt.hour
    df['is_rush_hour'] = (
        (df['hour'].isin([7, 8, 9])) | 
        (df['hour'].isin([17, 18, 19, 20, 21, 22]))
    ).astype(int)
    df['is_weekend'] = (df['datetime'].dt.weekday >= 5).astype(int)
    
    # Season (based on Karachi climate)
    month = df['datetime'].dt.month
    df['season'] = np.select([
        month.isin([12, 1, 2]),  # Winter
        month.isin([3, 4, 5]),   # Spring  
        month.isin([6, 7, 8]),   # Summer
        month.isin([9, 10, 11])  # Monsoon
    ], [0, 1, 2, 3], default=0)
    
    # Lag features
    df['pm2_5_nowcast_lag_1h'] = df['pm2_5_nowcast'].shift(1)
    df['no2_ppb_lag_1h'] = df['no2_ppb'].shift(1)
    
    # Add datetime_id as Unix timestamp for online compatibility
    df['datetime_id'] = df['datetime'].astype('int64') // 10**9
    
    # Select only production features
    feature_cols = list(PRODUCTION_FEATURES_SCHEMA.keys())
    df_features = df[feature_cols].copy()
    
    print(f"✅ Feature engineering complete. Shape: {df_features.shape}")
    return df_features

# Duplicate functions removed - using consolidated apply_all_feature_engineering function above

# ============================================================
# Data Quality and Imputation Functions  
# ============================================================
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
                hist_data = get_historical_data_for_imputation(fs, feature)
                
                if len(hist_data) > 0:
                    impute_value = hist_data.median()
                    df.loc[missing_mask, feature] = impute_value
                else:
                    df[feature] = df[feature].fillna(method='ffill').fillna(method='bfill')
    
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
    """Upload data to Hopsworks feature group using the simple, working approach from enhanced pipeline."""
    try:
        print(f"📤 Uploading {len(df)} records to Hopsworks...")
        
        # Get or create feature group using the simple approach
        fg = create_or_get_feature_group(fs, df)
        
        if fg is None:
            print("❌ Failed to get or create feature group")
            return False
        
        # Ensure datetime is timezone-aware (keep this from unified pipeline)
        if df['datetime'].dt.tz is None:
            df['datetime'] = df['datetime'].dt.tz_localize(PKT)
        
        # Use the simple insert approach from enhanced pipeline
        fg.insert(df, write_options={"wait_for_job": True})
        
        print(f"✅ Successfully uploaded {len(df)} records to Hopsworks")
        print(f"   📊 Feature Group: {FEATURE_GROUP_NAME}")
        print(f"   📈 Features: {len(df.columns) - 1}")  # Exclude datetime
        print(f"   📅 Time Range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        return True
        
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
        start_date = end_date - relativedelta(months=12)
        
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
    Hourly pipeline: Fetch recent data and append to existing feature group.
    If feature group doesn't exist, run backfill first.
    """
    print("\n" + "="*60)
    print("⏰ STARTING HOURLY PIPELINE")
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
        
        # Fetch last 19 hours: 16 hours for proper averages + 3 hours for lag calculation
        # This ensures CO 8hr avg and PM NowCast have sufficient data
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(hours=19)
        
        print(f"📅 Fetching last 19 hours: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
        print("   (16 hours for proper averages + 3 hours for lag calculation)")
        
        # Fetch data with retry
        json_data = fetch_aqi_data_with_retry(LAT, LON, start_date, end_date)
        if not json_data:
            print("❌ Failed to fetch hourly data")
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
            print(f"\n✅ HOURLY PIPELINE COMPLETED SUCCESSFULLY")
            print(f"📊 Added {len(df_latest)} new record")
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

