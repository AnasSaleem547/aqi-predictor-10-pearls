#!/usr/bin/env python3
"""
Debug script to check what data is actually in the Hopsworks feature group
and diagnose why the preview might be empty.
"""

import os
import pandas as pd
from datetime import datetime, timezone
import hopsworks
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("🔍 DEBUGGING HOPSWORKS FEATURE GROUP DATA")
    print("=" * 60)
    
    try:
        # Connect to Hopsworks
        print("🔗 Connecting to Hopsworks...")
        project = hopsworks.login()
        fs = project.get_feature_store()
        print(f"✅ Connected to project: {project.name}")
        
        # Get the feature group
        print("\n📊 Getting feature group...")
        fg = fs.get_feature_group("karachifeatures", version=1)
        
        if fg is None:
            print("❌ Feature group not found!")
            return
            
        print(f"✅ Found feature group: {fg.name}")
        print(f"📈 Version: {fg.version}")
        print(f"🏪 Type: {'Online' if fg.online_enabled else 'Offline'}")
        
        # Try to read data
        print("\n📖 Reading data from feature group...")
        try:
            # Try to read all data
            df = fg.read()
            print(f"✅ Successfully read data. Shape: {df.shape}")
            
            if len(df) == 0:
                print("⚠️ Feature group exists but contains no data!")
            else:
                print(f"📊 Data shape: {df.shape}")
                print(f"🕐 Date range: {df['datetime'].min()} to {df['datetime'].max()}")
                print("\n📋 Column info:")
                for col in df.columns:
                    non_null = df[col].notna().sum()
                    print(f"  {col}: {non_null}/{len(df)} non-null values")
                
                print("\n🔍 Sample data (first 3 rows):")
                print(df.head(3).to_string())
                
                print("\n🔍 Data types:")
                print(df.dtypes)
                
        except Exception as read_error:
            print(f"❌ Error reading data: {read_error}")
            
            # Try alternative read methods
            print("\n🔄 Trying alternative read methods...")
            try:
                # Try reading with limit
                df_limited = fg.read(limit=10)
                print(f"✅ Limited read successful. Shape: {df_limited.shape}")
                if len(df_limited) > 0:
                    print(df_limited.head())
            except Exception as e2:
                print(f"❌ Limited read also failed: {e2}")
        
        # Check feature group statistics
        print(f"\n📊 Feature group info:")
        try:
            print(f"  Primary key: {fg.primary_key}")
            print(f"  Event time: {fg.event_time}")
            print(f"  Features: {len(fg.features)} total")
            for feature in fg.features[:5]:  # Show first 5 features
                print(f"    - {feature.name}: {feature.type}")
            if len(fg.features) > 5:
                print(f"    ... and {len(fg.features) - 5} more")
                
        except Exception as info_error:
            print(f"❌ Error getting feature group info: {info_error}")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()