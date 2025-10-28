#!/usr/bin/env python3
"""
Test Gap Detection Logic
========================

This script demonstrates how the enhanced hourly pipeline would detect
and fill gaps in your data, using your current scenario where you have
records at '28/10/2025 3:00' and '28/10/2025 9:00'.
"""

import pandas as pd
from datetime import datetime, timedelta
import pytz

# Define PKT timezone
PKT = pytz.timezone('Asia/Karachi')

def simulate_gap_detection():
    """Simulate the gap detection logic with your current data."""
    
    print("🧪 SIMULATING GAP DETECTION LOGIC")
    print("=" * 50)
    
    # Your current data scenario - showing the REAL gap issue
    existing_records = [
        datetime(2025, 10, 28, 3, 0),  # 3:00 AM
        datetime(2025, 10, 28, 9, 0),  # 9:00 AM
    ]
    
    print("📊 Current records in Hopsworks:")
    for record in existing_records:
        print(f"   📅 {record.strftime('%Y-%m-%d %H:%M')} PKT")
    
    print(f"\n🚨 NOTICE: There's a 6-hour gap between 3:00 AM and 9:00 AM!")
    print(f"   Missing hours: 4:00, 5:00, 6:00, 7:00, 8:00")
    
    # Get the last record (most recent)
    last_timestamp = max(existing_records)
    print(f"\n📅 Last record timestamp: {last_timestamp.strftime('%Y-%m-%d %H:%M')} PKT")
    
    # Current time (simulated as 10:30 AM for demonstration)
    current_time = datetime(2025, 10, 28, 10, 30)
    current_hour = current_time.replace(minute=0, second=0, microsecond=0)
    print(f"🕐 Current time: {current_time.strftime('%Y-%m-%d %H:%M')} PKT")
    print(f"🕐 Current hour (rounded): {current_hour.strftime('%Y-%m-%d %H:%M')} PKT")
    
    # Calculate the next expected hour after last record
    next_expected_hour = (last_timestamp + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    print(f"⏭️  Next expected hour: {next_expected_hour.strftime('%Y-%m-%d %H:%M')} PKT")
    
    # Check for gap
    if next_expected_hour >= current_hour:
        print("✅ No gap detected - data is up to date")
        return []
    else:
        # Calculate gap
        gap_hours = int((current_hour - next_expected_hour).total_seconds() / 3600)
        print(f"\n🔍 GAP DETECTED!")
        print(f"   ⏰ Gap duration: {gap_hours} hours")
        print(f"   📅 Gap start: {next_expected_hour.strftime('%Y-%m-%d %H:%M')} PKT")
        print(f"   📅 Gap end: {current_hour.strftime('%Y-%m-%d %H:%M')} PKT")
        
        # Generate list of missing hours
        missing_hours = []
        current_missing = next_expected_hour
        while current_missing < current_hour:
            missing_hours.append(current_missing)
            current_missing += timedelta(hours=1)
        
        print(f"\n🔧 MISSING HOURS TO FILL:")
        for i, hour in enumerate(missing_hours, 1):
            print(f"   {i:2d}. {hour.strftime('%Y-%m-%d %H:%M')} PKT")
        
        return missing_hours

def simulate_complete_gap_analysis():
    """Show the complete gap analysis including the historical gap."""
    
    print("\n" + "=" * 60)
    print("🔍 COMPLETE GAP ANALYSIS")
    print("=" * 60)
    
    # Your actual data
    existing_records = [
        datetime(2025, 10, 28, 3, 0),  # 3:00 AM
        datetime(2025, 10, 28, 9, 0),  # 9:00 AM
    ]
    
    # What SHOULD be there for continuous hourly data
    expected_start = datetime(2025, 10, 28, 3, 0)
    current_time = datetime(2025, 10, 28, 10, 30)
    current_hour = current_time.replace(minute=0, second=0, microsecond=0)
    
    # Generate all expected hours
    all_expected_hours = []
    current_expected = expected_start
    while current_expected <= current_hour:
        all_expected_hours.append(current_expected)
        current_expected += timedelta(hours=1)
    
    print("📊 WHAT YOUR DATA SHOULD LOOK LIKE:")
    missing_count = 0
    for hour in all_expected_hours:
        if hour in existing_records:
            print(f"   ✅ {hour.strftime('%Y-%m-%d %H:%M')} PKT - EXISTS")
        else:
            print(f"   ❌ {hour.strftime('%Y-%m-%d %H:%M')} PKT - MISSING")
            missing_count += 1
    
    print(f"\n📈 SUMMARY:")
    print(f"   📊 Total expected records: {len(all_expected_hours)}")
    print(f"   ✅ Existing records: {len(existing_records)}")
    print(f"   ❌ Missing records: {missing_count}")
    print(f"   📉 Data completeness: {len(existing_records)/len(all_expected_hours)*100:.1f}%")
    
    return [h for h in all_expected_hours if h not in existing_records]

def simulate_what_would_happen():
    """Simulate what the enhanced pipeline would do."""
    
    print("\n" + "=" * 60)
    print("🚀 WHAT THE ENHANCED PIPELINE WOULD DO")
    print("=" * 60)
    
    missing_hours = simulate_gap_detection()
    
    if missing_hours:
        print(f"\n📡 The pipeline would:")
        print(f"   1. 🔍 Detect {len(missing_hours)} missing hours")
        print(f"   2. 📦 Process data in chunks (max 48 hours per chunk)")
        print(f"   3. 🔄 Retry failed chunks up to 3 times each")
        print(f"   4. 📊 Fetch data with 16-hour buffer for proper averages")
        print(f"   5. 🧮 Apply feature engineering to all fetched data")
        print(f"   6. ✂️  Filter to only the gap period")
        print(f"   7. 🔧 Impute missing values using historical data")
        print(f"   8. ✅ Validate data quality")
        print(f"   9. 📤 Upload {len(missing_hours)} records to Hopsworks")
        print(f"  10. 📡 Continue with normal current-hour processing")
        
        print(f"\n📈 EXPECTED RESULT:")
        print(f"   📊 Total new records: {len(missing_hours) + 1}")
        print(f"   🔧 Gap records: {len(missing_hours)}")
        print(f"   📡 Current hour: 1")
        print(f"   📅 Data continuity: RESTORED ✅")
        
        print(f"\n🎯 YOUR DATA AFTER PIPELINE RUN:")
        all_hours = [
            datetime(2025, 10, 28, 3, 0),   # Existing
            datetime(2025, 10, 28, 9, 0),   # Existing
        ] + missing_hours + [datetime(2025, 10, 28, 10, 0)]  # New current hour
        
        all_hours.sort()
        for i, hour in enumerate(all_hours, 1):
            status = "📊 Existing" if hour in [datetime(2025, 10, 28, 3, 0), datetime(2025, 10, 28, 9, 0)] else "🆕 New"
            print(f"   {i:2d}. {hour.strftime('%Y-%m-%d %H:%M')} PKT - {status}")
    else:
        print("✅ No gaps to fill - would proceed with normal hourly update")

if __name__ == "__main__":
    # First show the complete gap analysis
    all_missing = simulate_complete_gap_analysis()
    
    # Then show what the current pipeline logic would detect
    simulate_what_would_happen()
    
    print(f"\n💡 KEY BENEFITS:")
    print(f"   🔄 Automatic gap recovery without manual intervention")
    print(f"   🛡️  Robust error handling for network issues")
    print(f"   📊 Maintains data continuity for ML training")
    print(f"   ⚡ Efficient chunked processing for large gaps")
    print(f"   🎯 Perfect for automated cron jobs")
    
    print(f"\n🚨 IMPORTANT NOTE ABOUT YOUR CURRENT SITUATION:")
    print(f"   The enhanced pipeline only fills gaps AFTER the last record.")
    print(f"   Historical gaps (like 4:00-8:00 AM) would need manual backfill.")
    print(f"   Consider running: python unified_aqi_hopsworks_pipeline.py --backfill")
    print(f"   Or manually fetch those specific hours if needed.")