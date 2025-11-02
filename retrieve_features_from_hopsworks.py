#!/usr/bin/env python3
"""
Retrieve Features from Hopsworks Feature Store
==============================================

This script retrieves the Karachi AQI features that were uploaded to Hopsworks
by the enhanced_aqi_pipeline.py script.

Author: Anas Saleem
Institution: FAST NUCES
"""

import os
import pandas as pd
import pytz
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Load environment variables
load_dotenv()

# Hopsworks imports with error handling
try:
    import hopsworks
    HOPSWORKS_AVAILABLE = True
except ImportError:
    HOPSWORKS_AVAILABLE = False
    print("❌ Hopsworks SDK not available. Install with: pip install hopsworks")
    exit(1)

# Configuration
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
FEATURE_GROUP_NAME = "karachifeatures"
FEATURE_GROUP_VERSION = 1  # Changed from 3 to 1 (the version that actually exists)

# Console for rich output
console = Console(force_terminal=True, width=200)

def connect_to_hopsworks():
    """Connect to Hopsworks Feature Store."""
    if not HOPSWORKS_PROJECT or not HOPSWORKS_API_KEY:
        console.print("❌ [bold red]Hopsworks credentials not found in .env file[/bold red]")
        return None, None
    
    try:
        console.print(f"🔗 [bold cyan]Connecting to Hopsworks project: {HOPSWORKS_PROJECT}[/bold cyan]")
        
        # Connect to Hopsworks
        project = hopsworks.login(
            project=HOPSWORKS_PROJECT,
            api_key_value=HOPSWORKS_API_KEY
        )
        
        # Get feature store
        fs = project.get_feature_store()
        
        console.print("✅ [bold green]Successfully connected to Hopsworks[/bold green]")
        return project, fs
        
    except Exception as e:
        console.print(f"❌ [bold red]Failed to connect to Hopsworks: {str(e)}[/bold red]")
        return None, None

def convert_utc_to_pkt_and_fix_features(df):
    """
    Convert UTC datetime to PKT and recalculate time-based features.
    
    This function addresses the issue where:
    1. Hopsworks stores datetime in UTC but we need PKT for analysis
    2. Hour-based features become inconsistent due to timezone differences
    
    Parameters:
    - df: DataFrame with UTC datetime and potentially incorrect hour-based features
    
    Returns:
    - DataFrame with PKT datetime and corrected time-based features
    """
    if df is None or df.empty:
        return df
    
    console.print(f"\n🕐 [bold cyan]Converting UTC to PKT and fixing time-based features...[/bold cyan]")
    
    # Define PKT timezone
    PKT = pytz.timezone('Asia/Karachi')
    
    # Convert datetime from UTC to PKT
    if 'datetime' in df.columns:
        # Ensure datetime is timezone-aware (UTC)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        
        # Convert to PKT
        df['datetime'] = df['datetime'].dt.tz_convert(PKT)
        
        # Extract correct PKT hour
        df['hour'] = df['datetime'].dt.hour
        
        console.print(f"   ✅ Converted datetime from UTC to PKT")
        console.print(f"   ✅ Recalculated hour field based on PKT")
        
        # Recalculate other time-based features if they exist
        if 'is_rush_hour' in df.columns:
            # Rush hours: 7-10 AM and 5-8 PM PKT
            df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 10)) | ((df['hour'] >= 17) & (df['hour'] <= 20))
            df['is_rush_hour'] = df['is_rush_hour'].astype(int)
            console.print(f"   ✅ Recalculated is_rush_hour based on PKT")
        
        if 'is_weekend' in df.columns:
            # Weekend: Saturday (5) and Sunday (6)
            df['is_weekend'] = df['datetime'].dt.dayofweek.isin([5, 6]).astype(int)
            console.print(f"   ✅ Recalculated is_weekend based on PKT")
        
        # Convert back to timezone-naive for consistency with local analysis
        df['datetime'] = df['datetime'].dt.tz_localize(None)
        
        console.print(f"   ✅ Converted to timezone-naive PKT for local analysis")
        console.print(f"   📅 Final datetime format: PKT timezone-naive (no timezone suffix)")
    
    return df

def retrieve_features(fs, limit=None):
    """
    Retrieve features from the Hopsworks feature group.
    
    Parameters:
    - fs: Hopsworks feature store connection
    - limit: Number of records to retrieve (None for all)
    
    Returns:
    - DataFrame with retrieved features
    """
    try:
        console.print(f"\n📥 [bold cyan]Retrieving features from: {FEATURE_GROUP_NAME} (v{FEATURE_GROUP_VERSION})[/bold cyan]")
        
        # Get the feature group
        feature_group = fs.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION
        )
        
        console.print(f"   📊 Feature Group: {feature_group.name}")
        console.print(f"   📈 Version: {feature_group.version}")
        console.print(f"   📝 Description: {feature_group.description}")
        
        # Create query to select all features
        query = feature_group.select_all()
        
        # Read data with optional limit
        if limit:
            console.print(f"   📋 Retrieving latest {limit} records...")
            df = query.read(limit=limit)
        else:
            console.print(f"   📋 Retrieving all records...")
            df = query.read()
        
        # Sort by datetime to ensure chronological order
        if 'datetime' in df.columns:
            console.print(f"   🔄 Sorting data by datetime for chronological order...")
            df = df.sort_values('datetime').reset_index(drop=True)
        
        # Convert UTC to PKT and fix time-based features
        df = convert_utc_to_pkt_and_fix_features(df)
        
        console.print(f"✅ [bold green]Successfully retrieved {len(df)} records[/bold green]")
        console.print(f"   📊 Shape: {df.shape}")
        console.print(f"   📅 Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        return df
        
    except Exception as e:
        console.print(f"❌ [bold red]Failed to retrieve features: {str(e)}[/bold red]")
        return None

def display_feature_info(df):
    """Display information about the retrieved features."""
    console.print(f"\n📊 [bold cyan]FEATURE INFORMATION[/bold cyan]")
    console.print("=" * 80)
    
    # Basic statistics
    console.print(f"📈 Total Records: {len(df):,}")
    console.print(f"📋 Total Features: {len(df.columns)}")
    console.print(f"📅 Time Range: {df['datetime'].min()} → {df['datetime'].max()}")
    console.print(f"⏱️  Duration: {df['datetime'].max() - df['datetime'].min()}")
    
    # Feature list
    console.print(f"\n🏷️  [bold cyan]Available Features:[/bold cyan]")
    for i, col in enumerate(df.columns, 1):
        dtype = str(df[col].dtype)
        non_null = df[col].count()
        console.print(f"   {i:2d}. {col:<25} ({dtype:<10}) - {non_null:,} non-null values")
    
    # Target variable summary
    if 'aqi_epa_calc' in df.columns:
        aqi_stats = df['aqi_epa_calc'].describe()
        console.print(f"\n🎯 [bold cyan]Target Variable (AQI) Summary:[/bold cyan]")
        console.print(f"   Count: {aqi_stats['count']:.0f}")
        console.print(f"   Mean:  {aqi_stats['mean']:.1f}")
        console.print(f"   Std:   {aqi_stats['std']:.1f}")
        console.print(f"   Min:   {aqi_stats['min']:.1f}")
        console.print(f"   Max:   {aqi_stats['max']:.1f}")

def display_sample_data(df, n_records=10):
    """Display sample data using Rich table."""
    console.print(f"\n📋 [bold cyan]Sample Data (Latest {n_records} Records):[/bold cyan]")
    
    # Create Rich table
    table = Table(show_header=True, header_style="bold cyan", expand=True)
    
    # Add columns (limit to key features for readability)
    key_columns = ['datetime', 'aqi_epa_calc', 'pm2_5_nowcast', 'pm10_nowcast', 
                   'co_ppm_8hr_avg', 'no2_ppb', 'so2_ppb', 'hour', 'is_rush_hour', 'season']
    
    display_columns = [col for col in key_columns if col in df.columns]
    
    for col in display_columns:
        table.add_column(col, justify="right" if pd.api.types.is_numeric_dtype(df[col]) else "left")
    
    # Add rows (show latest N records)
    sample_df = df.tail(n_records)
    for _, row in sample_df.iterrows():
        values = []
        for col in display_columns:
            val = row[col]
            if pd.isna(val):
                values.append("-")
            elif isinstance(val, float):
                values.append(f"{val:.2f}")
            else:
                values.append(str(val))
        table.add_row(*values)
    
    console.print(table)

def save_retrieved_data(df, filename="retrieved_karachi_aqi_features12.csv"):
    """Save retrieved data to local CSV file with proper datetime formatting."""
    try:
        # Create a copy for saving to avoid modifying the original
        df_save = df.copy()
        
        # Ensure datetime is saved in a clear PKT format
        if 'datetime' in df_save.columns:
            # Add a note that this is PKT time in the filename or add a comment
            console.print(f"   📅 Saving datetime as timezone-naive PKT (Pakistan Standard Time)")
        
        df_save.to_csv(filename, index=False)
        console.print(f"\n💾 [bold green]Saved retrieved data to: {filename}[/bold green]")
        console.print(f"   📊 Records: {len(df_save):,}")
        console.print(f"   📋 Features: {len(df_save.columns)}")
        console.print(f"   ⚠️  Note: Datetime column is in PKT (Pakistan Standard Time)")
        return True
    except Exception as e:
        console.print(f"❌ [bold red]Failed to save data: {str(e)}[/bold red]")
        return False

def main():
    """Main function to retrieve and display Hopsworks features."""
    console.print("\n🔍 [bold green]RETRIEVE KARACHI AQI FEATURES FROM HOPSWORKS[/bold green]")
    console.print("=" * 80)
    
    try:
        # Step 1: Connect to Hopsworks
        project, fs = connect_to_hopsworks()
        if not project or not fs:
            console.print("❌ [bold red]Cannot proceed without Hopsworks connection[/bold red]")
            return None
        
        # Step 2: Retrieve features
        # You can specify a limit here, e.g., retrieve_features(fs, limit=100)
        df = retrieve_features(fs)
        if df is None or df.empty:
            console.print("❌ [bold red]No data retrieved[/bold red]")
            return None
        
        # Step 3: Display feature information
        display_feature_info(df)
        
        # Step 4: Display sample data
        display_sample_data(df, n_records=10)
        
        # Step 5: Save to local file
        save_retrieved_data(df)
        
        console.print("\n🎉 [bold green]FEATURE RETRIEVAL COMPLETED SUCCESSFULLY![/bold green]")
        console.print("=" * 80)
        
        return df
        
    except Exception as e:
        console.print(f"\n❌ [bold red]Feature retrieval failed: {str(e)}[/bold red]")
        import traceback
        console.print(f"📋 [bold red]Traceback:[/bold red]")
        console.print(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Retrieve features
    retrieved_df = main()
    
    if retrieved_df is not None:
        console.print(f"\n📊 [bold cyan]Retrieved Dataset Shape: {retrieved_df.shape}[/bold cyan]")
        console.print(f"📈 Features: {retrieved_df.shape[1] - 1}")  # Exclude datetime
        console.print(f"📋 Records: {retrieved_df.shape[0]:,}")
        console.print("\n✅ [bold green]Data ready for analysis and ML training![/bold green]")
    else:
        console.print("\n❌ [bold red]Feature retrieval failed.[/bold red]")